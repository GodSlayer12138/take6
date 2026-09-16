/* eslint-env worker */
import { chooseCard } from './ai-core.js'

self.onmessage = (event) => {
  try {
    self.postMessage({ ok: true, ...chooseCard(event.data, event.data.strategy || 'neural_hybrid') })
  } catch (error) {
    self.postMessage({ ok: false, message: error instanceof Error ? error.message : String(error) })
  }
}
