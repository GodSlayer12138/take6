// Presentation and routing live outside the frozen research engine.
export const STATIC_AI = import.meta.env?.MODE === 'static'

export function modelFor({ playerCount, deckSize, deckMode }, staticMode = STATIC_AI) {
  const classic = deckSize === 104 || (deckSize == null && deckMode === 'classic')
  if (!staticMode && classic && [3, 4].includes(playerCount)) return { id: 'distill2048-specialist2048', label: '正式冠军 · 2048', remote: true, certified: true }
  if (classic && playerCount === 2) return { id: 'v6', label: '二人专项 · V6', remote: false, certified: true }
  return { id: 'neural_hybrid', label: '浏览器混合 AI', remote: false, certified: false }
}

export function publicPayload(observation) {
  // Never forward players, other hands, opponents' identities or pending cards.
  const { hand, rows, seenCards, playerCount, deckSize, scores, seed } = observation
  return { hand: [...hand], rows: rows.map(row => [...row]), seenCards: [...seenCards], playerCount, deckSize, scores: [...scores], seed }
}

export async function chooseWebCard(observation, strategy = 'strongest', options = {}) {
  const route = modelFor(observation, options.staticMode ?? STATIC_AI)
  if (strategy === 'strongest' && route.remote) {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 60000)
    try {
      const response = await (options.fetch || globalThis.fetch)('/api/ai/choose', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(publicPayload(observation)), signal: controller.signal,
      })
      let result
      try { result = await response.json() } catch { throw new Error('冠军服务未就绪，请用 npm run dev 或 npm start 启动完整应用') }
      if (!response.ok) throw new Error(result.error || '冠军服务不可用，请检查本机服务')
      if (result.model !== route.id || !observation.hand.includes(result.card) || result.worlds !== 2048) throw new Error('冠军响应与冻结模型不一致')
      return result
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('冠军推演超时，请重试当前回合')
      if (error instanceof TypeError) throw new Error('无法连接本机冠军服务，请确认完整应用正在运行')
      throw error
    } finally { clearTimeout(timeout) }
  }
  if (strategy === 'strongest' && route.id === 'v6') {
    const { chooseV6 } = await import('./v6-runtime.js')
    return { ...chooseV6(observation), model: 'v6' }
  }
  const { chooseCard } = await import('../game/ai-core.js')
  return { ...chooseCard(observation, strategy === 'strongest' ? 'neural_hybrid' : strategy), model: strategy === 'strongest' ? 'neural_hybrid' : strategy }
}
