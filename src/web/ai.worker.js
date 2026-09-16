import { chooseWebCard } from './ai-router.js'

self.onmessage = async ({ data }) => {
  try {
    self.postMessage({ ok: true, ...await chooseWebCard(data, data.strategy || 'strongest') })
  } catch (error) {
    self.postMessage({ ok: false, message: error.message || String(error) })
  }
}
