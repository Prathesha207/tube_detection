const fs = require('fs');
const path = require('path');

console.log('[patch] Running automatic compatibility fixes for electron-builder...');

function walkDir(dir, callback) {
  if (!fs.existsSync(dir)) return;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walkDir(fullPath, callback);
    } else if (entry.isFile() && (entry.name.endsWith('.js') || entry.name.endsWith('.ts'))) {
      callback(fullPath);
    }
  }
}

// 1. Recursively patch all files in app-builder-lib that reference ElectronDownloadCacheMode
const appBuilderDir = path.join(__dirname, 'node_modules', 'app-builder-lib');
if (fs.existsSync(appBuilderDir)) {
  walkDir(appBuilderDir, (filePath) => {
    try {
      let content = fs.readFileSync(filePath, 'utf8');
      let changed = false;

      // Fix ElectronDownloadCacheMode.ReadWrite
      if (content.includes('ElectronDownloadCacheMode.ReadWrite')) {
        content = content.replace(
          /(?:get_1\.)?ElectronDownloadCacheMode\.ReadWrite/g,
          '($& || undefined)'
        ).replace(
          'get_1.ElectronDownloadCacheMode.ReadWrite',
          '(get_1.ElectronDownloadCacheMode && get_1.ElectronDownloadCacheMode.ReadWrite) || undefined'
        ).replace(
          'ElectronDownloadCacheMode.ReadWrite',
          '(ElectronDownloadCacheMode && ElectronDownloadCacheMode.ReadWrite) || undefined'
        );
        changed = true;
      }

      // Fix ElectronDownloadCacheMode.WriteOnly
      if (content.includes('ElectronDownloadCacheMode.WriteOnly')) {
        content = content.replace(
          /cacheMode:\s*(?:get_1\.)?ElectronDownloadCacheMode\.WriteOnly/g,
          '...(typeof get_1 !== "undefined" && get_1.ElectronDownloadCacheMode ? { cacheMode: get_1.ElectronDownloadCacheMode.WriteOnly } : (typeof ElectronDownloadCacheMode !== "undefined" && ElectronDownloadCacheMode ? { cacheMode: ElectronDownloadCacheMode.WriteOnly } : {}))'
        );
        changed = true;
      }

      // Fix "Number(cacheOverride) in get_1.ElectronDownloadCacheMode"
      if (content.includes('in get_1.ElectronDownloadCacheMode')) {
        content = content.replace(
          /in\s+get_1\.ElectronDownloadCacheMode/g,
          'in (get_1.ElectronDownloadCacheMode || {})'
        );
        changed = true;
      }

      if (changed) {
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`[patch] Successfully patched: ${path.relative(__dirname, filePath)}`);
      }
    } catch (e) {
      console.warn(`[patch] Could not patch ${filePath}: ${e.message}`);
    }
  });
}

// 2. Fix @noble/hashes ESM incompatibility if version 2.x is present
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
