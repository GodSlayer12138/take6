import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { playArenaGame, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const modelPath = resolve(process.argv[2])
const games = Number(process.argv[3] || 200)
const referenceStrategy = process.argv[4]
const candidateStrategies = process.argv.slice(5)
const seedIndex = candidateStrategies.indexOf('--seed')
const seed = Number(seedIndex >= 0 ? candidateStrategies[seedIndex + 1] : process.env.NTW_SWEEP_SEED || 1910000000)
if (seedIndex >= 0) candidateStrategies.splice(seedIndex, 2)
const samples = Number(process.env.NTW_SWEEP_SAMPLES || 18)

if (!referenceStrategy || candidateStrategies.length === 0) {
  throw new Error('usage: node sweep-strategies.mjs MODEL GAMES REFERENCE CANDIDATE...')
}

const model = JSON.parse(readFileSync(modelPath, 'utf8'))
const opponents = ['champion', 'external_mcs', 'cautious', 'random']

function play(strategy, gameIndex) {
  const focalSeat = (5 - (gameIndex % 5)) % 5
  const strategies = Array(5)
  strategies[focalSeat] = strategy
  opponents.forEach((opponent, offset) => {
    strategies[(focalSeat + offset + 1) % 5] = opponent
  })
  const result = playArenaGame({
    playerCount: 5,
    deckMode: 'adaptive',
    strategies,
    seed: seed + gameIndex * 65537,
    championSamples: samples,
    legacySamples: Math.max(8, Math.round(samples / 2)),
  })
  const score = result.scores[focalSeat]
  const minimum = Math.min(...result.scores)
  const winnerCount = result.scores.filter((value) => value === minimum).length
  const win = score === minimum ? 1 / winnerCount : 0
  const lower = result.scores.filter((value) => value < score).length
  const tied = result.scores.filter((value) => value === score).length
  return { win, score, rank: 1 + lower + (tied - 1) / 2 }
}

function summarize(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length
  const variance = values.length > 1
    ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1)
    : 0
  const standardError = Math.sqrt(variance / values.length)
  return { mean, standardError, ci95: [mean - 1.96 * standardError, mean + 1.96 * standardError] }
}

setPolicyV2ModelForEvaluation(model)
const deltas = new Map(candidateStrategies.map((strategy) => [strategy, { win: [], score: [], rank: [] }]))
const referenceTotals = { win: 0, score: 0, rank: 0 }
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const reference = play(referenceStrategy, gameIndex)
  for (const key of Object.keys(referenceTotals)) referenceTotals[key] += reference[key]
  for (const strategy of candidateStrategies) {
    const candidate = play(strategy, gameIndex)
    const delta = deltas.get(strategy)
    for (const key of Object.keys(delta)) delta[key].push(candidate[key] - reference[key])
  }
  const reportEvery = Math.max(1, Math.floor(games / 10))
  if ((gameIndex + 1) % reportEvery === 0) {
    console.error(`[sweep] ${gameIndex + 1}/${games} games · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}

const results = candidateStrategies.map((strategy) => {
  const delta = Object.fromEntries(Object.entries(deltas.get(strategy)).map(([key, values]) => [key, summarize(values)]))
  delta.win.meanPP = delta.win.mean * 100
  delta.win.ci95PP = delta.win.ci95.map((value) => value * 100)
  return { strategy, delta }
}).sort((left, right) => right.delta.win.mean - left.delta.win.mean)

console.log(JSON.stringify({
  modelPath,
  games,
  seed,
  samples,
  referenceStrategy,
  reference: Object.fromEntries(Object.entries(referenceTotals).map(([key, value]) => [key, value / games])),
  results,
  seconds: (performance.now() - started) / 1000,
}))
