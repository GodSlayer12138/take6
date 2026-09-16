import { useEffect, useState } from 'react'
import { modelFor } from './ai-router.js'

export default function ModelStatus({ config }) {
  const route = modelFor(config)
  const [health, setHealth] = useState('checking')
  useEffect(() => {
    if (!route.remote) return undefined
    let active = true
    const controller = new AbortController()
    async function check() {
      try {
        const response = await fetch('/api/ai/health', { signal: controller.signal })
        const result = await response.json()
        if (active) setHealth(response.ok && result.status === 'ready' && result.policy === route.id ? 'ready' : 'unavailable')
      } catch { if (active) setHealth('unavailable') }
    }
    check()
    const timer = setInterval(check, 10000)
    return () => { active = false; controller.abort(); clearInterval(timer) }
  }, [route.remote, route.id])
  return <div className="model-status" role="status">
    <strong>{route.label}</strong>
    <span>{route.remote
      ? health === 'ready' ? '本机 GPU 已连接 · 固定 2,048 个模拟世界' : health === 'checking' ? '正在连接本机冠军…' : '冠军未连接，请使用完整应用启动命令'
      : route.id === 'v6' ? '二人专项 · 浏览器本地运行' : '此规则暂无正式专项冠军，使用经典 AI'}</span>
  </div>
}
