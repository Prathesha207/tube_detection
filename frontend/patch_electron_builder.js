const fs = require('fs');
const path = require('path');

console.log('[patch] Running automatic compatibility fixes for electron-builder...');

// 1. Fix @electron/get ElectronDownloadCacheMode error in app-builder-lib
const targetJs = path.join(__dirname, 'node_modules', 'app-builder-lib', 'out', 'util', 'electronGet.js');
if (fs.existsSync(targetJs)) {
  let content = fs.readFileSync(targetJs, 'utf8');
  let modified = false;

  if (content.includes('Number(cacheOverride) in get_1.ElectronDownloadCacheMode')) {
    content = content.replace(
      'Number(cacheOverride) in get_1.ElectronDownloadCacheMode',
      'get_1.ElectronDownloadCacheMode && Number(cacheOverride) in get_1.ElectronDownloadCacheMode'
    );
    modified = true;
  }

  if (content.includes('return get_1.ElectronDownloadCacheMode.ReadWrite;')) {
    content = content.replace(
      'return get_1.ElectronDownloadCacheMode.ReadWrite;',
      'return (get_1.ElectronDownloadCacheMode && get_1.ElectronDownloadCacheMode.ReadWrite) || undefined;'
    );
    modified = true;
  }

  if (content.includes('cacheMode: get_1.ElectronDownloadCacheMode.WriteOnly')) {
    content = content.replace(
      'cacheMode: get_1.ElectronDownloadCacheMode.WriteOnly',
      '...(get_1.ElectronDownloadCacheMode ? { cacheMode: get_1.ElectronDownloadCacheMode.WriteOnly } : {})'
    );
    modified = true;
  }

  if (modified) {
    fs.writeFileSync(targetJs, content, 'utf8');
    console.log('[patch] Successfully patched app-builder-lib/out/util/electronGet.js');
  } else {
    console.log('[patch] electronGet.js is already patched or up-to-date.');
  }
}

// 2. Fix @noble/hashes ESM incompatibility if version 2.x was installed
const noblePkg = path.join(__dirname, 'node_modules', '@noble', 'hashes', 'package.json');
const pkijsNoble = path.join(__dirname, 'node_modules', 'pkijs', 'node_modules', '@noble', 'hashes');
if (fs.existsSync(noblePkg) && fs.existsSync(pkijsNoble)) {
  try {
    const pkg = JSON.parse(fs.readFileSync(noblePkg, 'utf8'));
    if (pkg.type === 'module' || String(pkg.version).startsWith('2.')) {
      console.log('[patch] Replacing ESM @noble/hashes 2.x with CommonJS 1.8.0 from pkijs...');
      fs.cpSync(pkijsNoble, path.dirname(noblePkg), { recursive: true, force: true });
      console.log('[patch] Successfully replaced @noble/hashes with CommonJS 1.8.0');
    }
  } catch (e) {
    console.warn('[patch] Warning checking @noble/hashes:', e.message);
  }
}

console.log('[patch] Compatibility fixes complete.');
