# """
# embedder.py -- standalone deep appearance embedder for duck re-identification.

# Bolt-on, NOT a change to duck_analyzer.py. It reads the same config dict your
# analyzer already loads and gives you one stable feature vector per duck crop,
# so ID matching can lean on a learned embedding instead of an HS histogram --
# the fix for "similar-coloured ducks swap IDs".

#     from embedder import DeepEmbedder
#     emb = DeepEmbedder.from_config(cfg)          # None if use_embedder: false
#     v = emb.embed(crop)                          # L2-normalized np.float32, or None
#     s = DeepEmbedder.cosine(v1, v2)              # similarity in [0,1]

# Config keys (add to config.yaml):
#     use_embedder: true            # false -> from_config() returns None, no cost
#     embedder_backend: resnet      # "resnet" | "siglip"  (switchable)
#     embedder_device: 0            # optional; defaults to the analyzer's `device`
#     embedder_half: true           # optional; defaults to `use_half`

# Backends
# --------
# - resnet : torchvision ResNet50, penultimate 2048-d global-pool features.
#            Light, already have torch/torchvision. Good default.
# - siglip : transformers SigLIP image encoder (google/siglip-base-patch16-224).
#            Stronger, heavier, needs `pip install transformers`.

# Loaded ONCE. Reuse the instance for every frame -- do not construct per call.
# All failures are non-fatal: a load or an embed that raises returns None so the
# analyzer transparently falls back to its histogram signature.
# """

# import numpy as np

# try:
#     import cv2
# except Exception:
#     cv2 = None

# import torch


# class DeepEmbedder:
#     def __init__(self, backend="resnet", device="cpu", half=False,
#                  model_name=None):
#         self.backend = str(backend).lower()
#         self.device = device
#         self.half = bool(half) and str(device).startswith("cuda")
#         self.model = None
#         self._preprocess = None
#         self.dim = None
#         self._ok = False

#         if self.backend == "resnet":
#             self._init_resnet()
#         elif self.backend == "siglip":
#             self._init_siglip(model_name or "google/siglip-base-patch16-224")
#         else:
#             print(f"[EMB] [WARN] unknown embedder_backend '{backend}', "
#                   f"disabling embedder.")

#     # ---------------- construction from config ---------------- #
#     @classmethod
#     def from_config(cls, cfg):
#         """Return a DeepEmbedder, or None if use_embedder is false/absent.
#         Never raises -- on any load failure returns None and prints a warning,
#         so the caller just keeps using its histogram signature."""
#         if not bool(cfg.get("use_embedder", False)):
#             return None
#         backend = str(cfg.get("embedder_backend", "resnet")).lower()
#         device = cfg.get("embedder_device", cfg.get("device", 0))
#         device = cls._resolve_device(device)
#         half = bool(cfg.get("embedder_half", cfg.get("use_half", False)))
#         try:
#             emb = cls(backend=backend, device=device, half=half)
#             if not emb._ok:
#                 return None
#             print(f"[EMB] deep embedder ready: backend={emb.backend} "
#                   f"dim={emb.dim} device={device} half={emb.half}")
#             return emb
#         except Exception as e:
#             print(f"[EMB] [WARN] could not init embedder ({backend}): {e} "
#                   f"-- falling back to histogram signature.")
#             return None

#     @staticmethod
#     def _resolve_device(device):
#         if device is None:
#             return "cuda:0" if torch.cuda.is_available() else "cpu"
#         s = str(device).strip().lower()
#         if s == "cpu":
#             return "cpu"
#         if s.startswith("cuda"):
#             return s if torch.cuda.is_available() else "cpu"
#         if s.isdigit():
#             return f"cuda:{s}" if torch.cuda.is_available() else "cpu"
#         return "cuda:0" if torch.cuda.is_available() else "cpu"

#     # ---------------- backends ---------------- #
#     def _init_resnet(self):
#         try:
#             import torchvision
#             from torchvision import transforms
#             try:
#                 weights = torchvision.models.ResNet50_Weights.DEFAULT
#                 net = torchvision.models.resnet50(weights=weights)
#             except Exception:
#                 # older torchvision
#                 net = torchvision.models.resnet50(pretrained=True)
#             # drop the classifier -> 2048-d global-pooled features
#             net.fc = torch.nn.Identity()
#             net.eval().to(self.device)
#             if self.half:
#                 net.half()
#             self.model = net
#             self.dim = 2048
#             self._preprocess = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Resize((256, 128)),   # person-reID-ish crop ratio
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225]),
#             ])
#             self._warmup()
#             self._ok = True
#         except Exception as e:
#             print(f"[EMB] [WARN] resnet init failed: {e}")
#             self._ok = False

#     def _init_siglip(self, model_name):
#         try:
#             from transformers import AutoModel, AutoProcessor
#             self._hf_processor = AutoProcessor.from_pretrained(model_name)
#             model = AutoModel.from_pretrained(model_name)
#             self._vision = model.vision_model.eval().to(self.device)
#             if self.half:
#                 self._vision.half()
#             self.model = self._vision
#             # infer dim from config
#             self.dim = int(getattr(model.config.vision_config, "hidden_size", 768))
#             self._preprocess = None  # handled by hf processor in embed()
#             self._warmup()
#             self._ok = True
#         except Exception as e:
#             print(f"[EMB] [WARN] siglip init failed (need `pip install "
#                   f"transformers`?): {e}")
#             self._ok = False

#     def _warmup(self):
#         try:
#             dummy = np.zeros((64, 48, 3), dtype=np.uint8)
#             _ = self.embed(dummy)
#         except Exception:
#             pass

#     # ---------------- inference ---------------- #
#     @torch.no_grad()
#     def embed(self, crop):
#         """crop: HxWx3 BGR uint8 (as produced by get_crop). Returns an
#         L2-normalized np.float32 vector, or None on empty/failure."""
#         if not self._ok or crop is None or getattr(crop, "size", 0) == 0:
#             return None
#         try:
#             if self.backend == "resnet":
#                 rgb = crop[:, :, ::-1].copy() if cv2 is None else \
#                     cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 x = self._preprocess(rgb).unsqueeze(0).to(self.device)
#                 if self.half:
#                     x = x.half()
#                 feat = self.model(x)
#             else:  # siglip
#                 rgb = crop[:, :, ::-1].copy() if cv2 is None else \
#                     cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
#                 inputs = self._hf_processor(images=rgb, return_tensors="pt")
#                 pv = inputs["pixel_values"].to(self.device)
#                 if self.half:
#                     pv = pv.half()
#                 out = self.model(pixel_values=pv)
#                 feat = out.pooler_output if getattr(out, "pooler_output", None) \
#                     is not None else out.last_hidden_state.mean(dim=1)

#             v = feat.reshape(-1).float().cpu().numpy()
#             n = np.linalg.norm(v)
#             if n > 0:
#                 v = v / n
#             return v.astype(np.float32)
#         except Exception as e:
#             print(f"[EMB] [WARN] embed failed (non-fatal): {e}")
#             return None

#     # ---------------- similarity ---------------- #
#     @staticmethod
#     def cosine(a, b):
#         """Cosine similarity of two L2-normalized vectors, clamped to [0,1].

#         This returns the TRUE cosine (dot product of unit vectors), floored at
#         0. It is NOT remapped through (dot+1)/2 -- that remap squashes the
#         common case of non-negative feature vectors (ReLU-activated ResNet
#         features, colour histograms) into a narrow 0.5..1.0 band, which makes
#         appearance thresholds like swap_verify_min_app / rebind_min_app almost
#         impossible to reason about. With the true cosine, identical crops score
#         ~1.0, clearly-different crops score near 0, so a threshold such as 0.5
#         sits sensibly in the middle. Negative cosines (possible only for signed
#         embeddings) are floored to 0. Returns 0.0 if either vector is None."""
#         if a is None or b is None:
#             return 0.0
#         try:
#             d = float(np.dot(a, b))
#             return max(0.0, min(1.0, d))
#         except Exception:
#             return 0.0

#     @staticmethod
#     def to_blob(v):
#         """np.float32 vector -> bytes for SQLite BLOB storage (or None)."""
#         if v is None:
#             return None
#         return np.asarray(v, dtype=np.float32).tobytes()

#     @staticmethod
#     def from_blob(b):
#         """bytes -> np.float32 vector (or None)."""
#         if b is None:
#             return None
#         return np.frombuffer(b, dtype=np.float32)


"""
embedder.py -- standalone deep appearance embedder for duck re-identification.

Bolt-on, NOT a change to duck_analyzer.py. It reads the same config dict your
analyzer already loads and gives you one stable feature vector per duck crop,
so ID matching can lean on a learned embedding instead of an HS histogram --
the fix for "similar-coloured ducks swap IDs".

    from embedder import DeepEmbedder
    emb = DeepEmbedder.from_config(cfg)          # None if use_embedder: false
    v = emb.embed(crop)                          # L2-normalized np.float32, or None
    s = DeepEmbedder.cosine(v1, v2)              # similarity in [0,1]

Config keys (add to config.yaml):
    use_embedder: true            # false -> from_config() returns None, no cost
    embedder_backend: resnet      # "resnet" | "siglip"  (switchable)
    embedder_device: 0            # optional; defaults to the analyzer's `device`
    embedder_half: true           # optional; defaults to `use_half`

Backends
--------
- resnet : torchvision ResNet50, penultimate 2048-d global-pool features.
           Light, already have torch/torchvision. Good default.
- siglip : transformers SigLIP image encoder (google/siglip-base-patch16-224).
           Stronger, heavier, needs `pip install transformers`.

Loaded ONCE. Reuse the instance for every frame -- do not construct per call.
All failures are non-fatal: a load or an embed that raises returns None so the
analyzer transparently falls back to its histogram signature.
"""

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

import torch


class DeepEmbedder:
    def __init__(self, backend="resnet", device="cpu", half=False,
                 model_name=None):
        self.backend = str(backend).lower()
        self.device = device
        self.half = bool(half) and str(device).startswith("cuda")
        self.model = None
        self._preprocess = None
        self.dim = None
        self._ok = False

        if self.backend == "resnet":
            self._init_resnet()
        elif self.backend == "siglip":
            self._init_siglip(model_name or "google/siglip-base-patch16-224")
        else:
            print(f"[EMB] [WARN] unknown embedder_backend '{backend}', "
                  f"disabling embedder.")

    # ---------------- construction from config ---------------- #
    @classmethod
    def from_config(cls, cfg):
        """Return a DeepEmbedder, or None if use_embedder is false/absent.
        Never raises -- on any load failure returns None and prints a warning,
        so the caller just keeps using its histogram signature."""
        if not bool(cfg.get("use_embedder", False)):
            return None
        backend = str(cfg.get("embedder_backend", "resnet")).lower()
        device = cfg.get("embedder_device", cfg.get("device", 0))
        device = cls._resolve_device(device)
        half = bool(cfg.get("embedder_half", cfg.get("use_half", False)))
        try:
            emb = cls(backend=backend, device=device, half=half)
            if not emb._ok:
                return None
            print(f"[EMB] deep embedder ready: backend={emb.backend} "
                  f"dim={emb.dim} device={device} half={emb.half}")
            return emb
        except Exception as e:
            print(f"[EMB] [WARN] could not init embedder ({backend}): {e} "
                  f"-- falling back to histogram signature.")
            return None

    @staticmethod
    def _resolve_device(device):
        if device is None:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        s = str(device).strip().lower()
        if s == "cpu":
            return "cpu"
        if s.startswith("cuda"):
            return s if torch.cuda.is_available() else "cpu"
        if s.isdigit():
            return f"cuda:{s}" if torch.cuda.is_available() else "cpu"
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---------------- backends ---------------- #
    def _init_resnet(self):
        try:
            import torchvision
            try:
                weights = torchvision.models.ResNet50_Weights.DEFAULT
                net = torchvision.models.resnet50(weights=weights)
            except Exception:
                # older torchvision
                net = torchvision.models.resnet50(pretrained=True)
            # drop the classifier -> 2048-d global-pooled features
            net.fc = torch.nn.Identity()
            net.eval().to(self.device)
            if self.half:
                net.half()
            self.model = net
            self.dim = 2048
            # Precompute ImageNet normalization tensors directly on target GPU device
            dtype = torch.float16 if self.half else torch.float32
            self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=dtype).view(1, 3, 1, 1)
            self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=dtype).view(1, 3, 1, 1)
            self._warmup()
            self._ok = True
        except Exception as e:
            print(f"[EMB] [WARN] resnet init failed: {e}")
            self._ok = False

    def _init_siglip(self, model_name):
        try:
            from transformers import AutoModel, AutoProcessor
            self._hf_processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            self._vision = model.vision_model.eval().to(self.device)
            if self.half:
                self._vision.half()
            self.model = self._vision
            # infer dim from config
            self.dim = int(getattr(model.config.vision_config, "hidden_size", 768))
            self._preprocess = None  # handled by hf processor in embed()
            self._warmup()
            self._ok = True
        except Exception as e:
            print(f"[EMB] [WARN] siglip init failed (need `pip install "
                  f"transformers`?): {e}")
            self._ok = False

    def _warmup(self):
        try:
            dummy = np.zeros((64, 48, 3), dtype=np.uint8)
            _ = self.embed(dummy)
        except Exception:
            pass

    # ---------------- inference ---------------- #
    @torch.no_grad()
    def embed(self, crop):
        """crop: HxWx3 BGR uint8 (as produced by get_crop). Returns an
        L2-normalized np.float32 vector, or None on empty/failure."""
        if not self._ok or crop is None or getattr(crop, "size", 0) == 0:
            return None
        try:
            if self.backend == "resnet":
                # Fast C++ resize via OpenCV
                resized = cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                # Direct GPU tensor creation and vectorized GPU normalization
                x = torch.from_numpy(rgb).to(self.device, non_blocking=True).permute(2, 0, 1).unsqueeze(0)
                x = x.half() if self.half else x.float()
                x = x.div_(255.0).sub_(self._mean).div_(self._std)
                feat = self.model(x)
            else:  # siglip
                rgb = crop[:, :, ::-1].copy() if cv2 is None else \
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                inputs = self._hf_processor(images=rgb, return_tensors="pt")
                pv = inputs["pixel_values"].to(self.device)
                if self.half:
                    pv = pv.half()
                out = self.model(pixel_values=pv)
                feat = out.pooler_output if getattr(out, "pooler_output", None) \
                    is not None else out.last_hidden_state.mean(dim=1)

            v = feat.reshape(-1).float().detach().cpu().numpy()
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            return v.astype(np.float32)
        except Exception as e:
            print(f"[EMB] [WARN] embed failed (non-fatal): {e}")
            return None

    # ---------------- similarity ---------------- #
    @staticmethod
    def cosine(a, b):
        """Cosine similarity of two L2-normalized vectors, clamped to [0,1].

        This returns the TRUE cosine (dot product of unit vectors), floored at
        0. It is NOT remapped through (dot+1)/2 -- that remap squashes the
        common case of non-negative feature vectors (ReLU-activated ResNet
        features, colour histograms) into a narrow 0.5..1.0 band, which makes
        appearance thresholds like swap_verify_min_app / rebind_min_app almost
        impossible to reason about. With the true cosine, identical crops score
        ~1.0, clearly-different crops score near 0, so a threshold such as 0.5
        sits sensibly in the middle. Negative cosines (possible only for signed
        embeddings) are floored to 0. Returns 0.0 if either vector is None."""
        if a is None or b is None:
            return 0.0
        try:
            d = float(np.dot(a, b))
            return max(0.0, min(1.0, d))
        except Exception:
            return 0.0

    @staticmethod
    def to_blob(v):
        """np.float32 vector -> bytes for SQLite BLOB storage (or None)."""
        if v is None:
            return None
        return np.asarray(v, dtype=np.float32).tobytes()

    @staticmethod
    def from_blob(b):
        """bytes -> np.float32 vector (or None)."""
        if b is None:
            return None
        return np.frombuffer(b, dtype=np.float32)