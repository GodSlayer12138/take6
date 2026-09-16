import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const python = process.env.NTW_PYTHON || 'D:/Programs/miniconda3/envs/ntw-ai/python.exe'
const production = process.argv.includes('--production')
const port = Number(process.env.NTW_API_PORT || 8765)
if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('NTW_API_PORT must be between 1024 and 65535')
if (!existsSync(python)) throw new Error('Set NTW_PYTHON to a Python executable with PyTorch and CUDA support')
if (production && !existsSync(path.join(root, 'dist/index.html'))) throw new Error('Run npm run build first')

const children = new Set()
let stopping = false
function stop(code = 0) {
  if (stopping) return
  stopping = true
  for (const child of children) child.kill()
  process.exitCode = code
}
function launch(executable, args, extraEnv = {}) {
  const child = spawn(executable, args, { cwd: root, stdio: 'inherit', windowsHide: true, env: { ...process.env, ...extraEnv } })
  children.add(child)
  child.on('error', error => { console.error(error.message); stop(1) })
  child.on('exit', code => { children.delete(child); if (!stopping) stop(code || 0) })
  return child
}
process.on('SIGINT', () => stop())
process.on('SIGTERM', () => stop())
process.on('exit', () => { for (const child of children) child.kill() })

const backend = launch(python, ['-u', 'server/champion_api.py', '--port', String(port), ...(production ? ['--static'] : [])])
console.log('正在校验并加载正式冠军…')
let ready = false
for (let attempt = 0; attempt < 120 && !stopping; attempt++) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/ai/health`, { signal: AbortSignal.timeout(1500) })
    const status = await response.json()
    if (response.ok && status.app === 'ntw-champion' && status.pid === backend.pid) { ready = true; break }
  } catch { /* Wait for this backend's startup; do not reuse another process. */ }
  await new Promise(resolve => setTimeout(resolve, 500))
}
if (!ready) {
  console.error('冠军服务启动失败；请检查 CUDA 环境或端口占用。')
  stop(1)
} else if (production) {
  console.log(`游戏已启动：http://127.0.0.1:${port}`)
} else {
  launch(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1'], { NTW_API_PORT: String(port) })
}
