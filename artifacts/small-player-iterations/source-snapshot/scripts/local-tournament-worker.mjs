import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import { chooseCard, setPolicyV2ModelForEvaluation, FINAL_ADAPTIVE_STRATEGY } from '../src/game/ai-core.js'
import { chooseNeuralCard } from '../src/game/neural-features.js'
import { choosePolicyV2Card, chooseCompactPolicyV2Card, chooseCompactOpponentCard, evaluatePolicyV2Cards, evaluateCompactPolicyV2Cards, evaluateCompactOpponentCards, encodePolicyV2State, forwardLayers } from '../src/game/policy-v2.js'
import { resolveTurn } from '../src/game/engine.js'

const manifest = JSON.parse(readFileSync(process.argv[2], 'utf8'))
const entrants = new Map(manifest.entrants.map(e => [e.id, e]))
const cache = new Map()
function modelFor(entry) {
  if (!cache.has(entry.path)) {
    const model = JSON.parse(readFileSync(entry.path, 'utf8'))
    if (model.modelPaths) model.models = model.modelPaths.map(path => modelFor({ path: resolve(dirname(entry.path), path) }))
    cache.set(entry.path, model)
  }
  return cache.get(entry.path)
}
function ranked(observation, model) {
  if (model.format === 'ntw-policy-v2') return evaluatePolicyV2Cards(observation, model)
  if (model.format === 'ntw-policy-v2-compact') return evaluateCompactPolicyV2Cards(observation, model)
  if (model.format === 'ntw-counterfactual-ensemble') {
    const members = model.models.map(m => ranked(observation, m))
    const weights = model.weights || members.map(() => 1)
    return observation.hand.map(card => ({ card, utility: members.reduce((sum, entries, i) => sum + entries.findIndex(e => e.card === card) * Math.max(0, weights[i]), 0) }))
      .sort((a, b) => a.utility - b.utility || a.card - b.card)
  }
  if (model.format === 'ntw-counterfactual-gated') {
    const [a, b] = model.models.map(m => ranked(observation, m))
    if (observation.hand.length === 1) return a
    const clip = v => Math.max(-2, Math.min(2, v))
    const features = [...encodePolicyV2State(observation), a[0].card / 54, b[0].card / 54,
      clip((a[0].logit - a[1].logit) / 5), clip((b[0].logit - b[1].logit) / 5),
      Number(a[0].card === b[0].card), (a[0].card - b[0].card) / 54]
    return forwardLayers(features, model.gateLayers)[0] > 0 ? a : b
  }
  return evaluateCompactOpponentCards(observation, model)
}
function act({ id, observation }) {
  const entry = entrants.get(id)
  if (!entry) throw new Error(`Unknown entrant ${id}`)
  const start = performance.now()
  let card
  if (observation.hand.length === 1) card = observation.hand[0]
  else if (entry.kind === 'strategy') {
    setPolicyV2ModelForEvaluation(null)
    card = chooseCard(observation, entry.strategy === 'final' ? FINAL_ADAPTIVE_STRATEGY : entry.strategy).card
  } else {
    const model = modelFor(entry)
    if (model.format === 'ntw-neural-v1') card = chooseNeuralCard(observation, model).card
    else if (model.format === 'ntw-policy-v2') card = choosePolicyV2Card(observation, model).card
    else if (model.format === 'ntw-policy-v2-compact') card = chooseCompactPolicyV2Card(observation, model).card
    else if (model.format.startsWith('ntw-counterfactual-')) card = ranked(observation, model)[0].card
    else card = chooseCompactOpponentCard(observation, model)
  }
  if (!observation.hand.includes(card)) throw new Error(`Illegal card from ${id}`)
  return { card, seconds: (performance.now() - start) / 1000 }
}
for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
  try {
    const request = JSON.parse(line)
    let result
    if (request.type === 'actions') result = request.items.map(act)
    else if (request.type === 'resolve') result = resolveTurn({ ...request, humanPlayerId: -1 })
    else if (request.type === 'ping') result = { ready: true }
    else throw new Error('Unknown request')
    process.stdout.write(JSON.stringify({ result }) + '\n')
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: error.stack }) + '\n')
  }
}
