import re

with open('electron/main.js', 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(
    r'function getStaticPath\(\) \{.*?\n\}', 
    'function getStaticPath() {\n  return path.join(__dirname, "..", "dist")\n}', 
    code, flags=re.DOTALL
)

with open('electron/main.js', 'w', encoding='utf-8') as f:
    f.write(code)
print('Patched electron/main.js')
