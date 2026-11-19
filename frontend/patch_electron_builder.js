const fs = require('fs');
const path = require('path');

console.log('[patch] Running automatic compatibility fixes for electron-builder...');

// 1. Fix electronGet.js
const electronGetPath = path.join(__dirname, 'node_modules', 'app-builder-lib', 'out', 'util', 'electronGet.js');
if (fs.existsSync(electronGetPath)) {
  let content = fs.readFileSync(electronGetPath, 'utf8');

  // Replace entire resolveCacheMode function cleanly (idempotent, never duplicates)
  content = content.replace(
    /function resolveCacheMode\(\)[\s\S]*?\n\}/,
    'function resolveCacheMode() {\n    return 0;\n}'
  );

  // Replace retry downloadArtifact cacheMode
  content = content.replace(
    /filePath = await get\.downloadArtifact\(\{ \.\.\.configWithProgress[\s\S]*?\}\);/,
    'filePath = await get.downloadArtifact({ ...configWithProgress });'
  );

  fs.writeFileSync(electronGetPath, content, 'utf8');
  console.log('[patch] Successfully patched electronGet.js');
}

// 2. Fix binDownload.js
const binDownloadPath = path.join(__dirname, 'node_modules', 'app-builder-lib', 'out', 'binDownload.js');
if (fs.existsSync(binDownloadPath)) {
  let content = fs.readFileSync(binDownloadPath, 'utf8');
  content = content.replace(
    /cacheMode:[\s\S]*?,\n/,
    'cacheMode: 0,\n'
  );
  fs.writeFileSync(binDownloadPath, content, 'utf8');
  console.log('[patch] Successfully patched binDownload.js');
}

// 3. Fix @noble/hashes ESM incompatibility
const noblePkg = path.join(__dirname, 'node_modules', '@noble', 'hashes', 'package.json');
const pkijsNoble = path.join(__dirname, 'node_modules', 'pkijs', 'node_modules', '@noble', 'hashes');
if (fs.existsSync(noblePkg) && fs.existsSync(pkijsNoble)) {
  try {
    const pkg = JSON.parse(fs.readFileSync(noblePkg, 'utf8'));
    if (pkg.type === 'module' || String(pkg.version).startsWith('2.')) {
      fs.cpSync(pkijsNoble, path.dirname(noblePkg), { recursive: true, force: true });
      console.log('[patch] Replaced ESM @noble/hashes with CommonJS 1.8.0');
    }
  } catch (e) {
    console.warn('[patch] Warning checking @noble/hashes:', e.message);
  }
}

console.log('[patch] Compatibility fixes complete.');
