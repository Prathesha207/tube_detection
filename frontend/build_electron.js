const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const targetDir = path.join(__dirname, 'dist_app');
const tmpDir = path.join(targetDir, 'win-unpacked.tmp');
const destDir = path.join(targetDir, 'win-unpacked');

console.log("Starting Electron Build...");

for (let attempt = 1; attempt <= 3; attempt++) {
  try {
    if (fs.existsSync(tmpDir)) {
      try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) {}
    }
    if (fs.existsSync(destDir)) {
      try { fs.rmSync(destDir, { recursive: true, force: true }); } catch (e) {}
    }
    
    execSync('npx electron-builder --win', { stdio: 'inherit' });
    console.log("Build Successful!");
    break;
  } catch (err) {
    console.warn(`Build attempt ${attempt} failed: ${err.message}`);
    if (fs.existsSync(tmpDir) && !fs.existsSync(destDir)) {
      console.log("Attempting manual rename of win-unpacked.tmp -> win-unpacked...");
      try {
        execSync('powershell -Command "Start-Sleep -s 2"');
        fs.renameSync(tmpDir, destDir);
        console.log("Manual rename succeeded!");
        break;
      } catch (renameErr) {
        console.error("Manual rename failed:", renameErr.message);
      }
    }
    if (attempt === 3) {
      process.exit(1);
    }
  }
}

