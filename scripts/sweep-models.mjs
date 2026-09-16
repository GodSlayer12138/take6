import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { playArenaGame, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const referenceModelPath = resolve(process.argv[2])
const games = Number(process.argv[3] || 200)
const strategy = process.argv[4]
const remainingArgs = process.argv.slice(5)
const seedIndex = remainingArgs.indexOf('--seed')
const seed = Number(seedIndex >= 0 ? remainingArgs[seedIndex + 1] : process.env.NTW_SWEEP_SEED || 1980000000)
if (seedIndex >= 0) remainingArgs.splice(seedIndex, 2)
const candidateModelPaths = remainingArgs.map((path) => resolve(path))
const samples = Number(process.env.NTW_SWEEP_SAMPLES || 18)

if (!strategy || candidateModelPaths.length === 0) {
  throw new Error('usage: node sweep-models.mjs REFERENCE_MODEL GAMES STRATEGY CANDIDATE_MODEL...')
}

const referenceModel = JSON.parse(readFileSync(referenceModelPath, 'utf8'))
const candidateModels = candidateModelPaths.map((path) => JSON.parse(readFileSync(path, 'utf8')))
const opponents = ['champion', 'external_mcs', 'cautious', 'random']

function play(gameIndex) {
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

const deltas = candidateModelPaths.map(() => ({ win: [], score: [], rank: [] }))
const started = performance.now()
for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  setPolicyV2ModelForEvaluation(referenceModel)
  const reference = play(gameIndex)
  candidateModels.forEach((model, modelIndex) => {
    setPolicyV2ModelForEvaluation(model)
    const candidate = play(gameIndex)
    for (const key of Object.keys(deltas[modelIndex])) deltas[modelIndex][key].push(candidate[key] - reference[key])
  })
  const reportEvery = Math.max(1, Math.floor(games / 10))
  if ((gameIndex + 1) % reportEvery === 0) {
    console.error(`[model-sweep] ${gameIndex + 1}/${games} games · ${((performance.now() - started) / 1000).toFixed(1)}s`)
  }
}

const results = candidateModelPaths.map((path, index) => {
  const delta = Object.fromEntries(Object.entries(deltas[index]).map(([key, values]) => [key, summarize(values)]))
  delta.win.meanPP = delta.win.mean * 100
  delta.win.ci95PP = delta.win.ci95.map((value) => value * 100)
  return { modelPath: path, delta }
}).sort((left, right) => right.delta.win.mean - left.delta.win.mean)

console.log(JSON.stringify({ referenceModelPath, games, seed, samples, strategy, results, seconds: (performance.now() - started) / 1000 }))
