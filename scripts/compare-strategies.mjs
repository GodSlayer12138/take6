import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { playArenaGame, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const modelPath = resolve(process.argv[2])
const games = Number(process.argv[3] || 1000)
const referenceStrategy = process.argv[4] || 'neural_hybrid'
const candidateStrategy = process.argv[5]
const seed = Number(process.argv[6] || 370000000)
const samples = Number(process.argv[7] || 18)
const referenceModelPath = resolve(process.argv[8] || modelPath)
const candidateModelPath = resolve(process.argv[9] || modelPath)

if (!candidateStrategy) throw new Error('candidate strategy is required')
const referenceModel = JSON.parse(readFileSync(referenceModelPath, 'utf8'))
const candidateModel = JSON.parse(readFileSync(candidateModelPath, 'utf8'))

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

const reference = { win: [], score: [], rank: [] }
const candidate = { win: [], score: [], rank: [] }
const delta = { win: [], score: [], rank: [] }
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  setPolicyV2ModelForEvaluation(referenceModel)
  const referenceResult = play(referenceStrategy, gameIndex)
  setPolicyV2ModelForEvaluation(candidateModel)
  const candidateResult = play(candidateStrategy, gameIndex)
  for (const key of Object.keys(delta)) {
    reference[key].push(referenceResult[key])
    candidate[key].push(candidateResult[key])
    delta[key].push(candidateResult[key] - referenceResult[key])
  }
}

const referenceSummary = Object.fromEntries(Object.entries(reference).map(([key, values]) => [key, summarize(values).mean]))
const candidateSummary = Object.fromEntries(Object.entries(candidate).map(([key, values]) => [key, summarize(values).mean]))
const deltaSummary = Object.fromEntries(Object.entries(delta).map(([key, values]) => [key, summarize(values)]))
deltaSummary.win.meanPP = deltaSummary.win.mean * 100
deltaSummary.win.ci95PP = deltaSummary.win.ci95.map((value) => value * 100)

console.log(JSON.stringify({
  modelPath,
  referenceModelPath,
  candidateModelPath,
  games,
  seed,
  samples,
  referenceStrategy,
  candidateStrategy,
  reference: referenceSummary,
  candidate: candidateSummary,
  delta: deltaSummary,
  seconds: (performance.now() - started) / 1000,
}))
