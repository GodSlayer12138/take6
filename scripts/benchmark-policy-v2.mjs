import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { playArenaGame, setPolicyV2ModelForEvaluation } from '../src/game/ai-core.js'

const modelPath = resolve(process.argv[2])
const games = Number(process.argv[3] || 200)
const strategy = process.argv[4] || 'neural_hybrid_v2_k70'
const seed = Number(process.argv[5] || 370000000)
const samples = Number(process.argv[6] || 18)
const pool = [strategy, 'champion', 'external_mcs', 'cautious', 'random']
setPolicyV2ModelForEvaluation(JSON.parse(readFileSync(modelPath, 'utf8')))

const stats = Object.fromEntries(pool.map((id) => [id, { wins: 0, score: 0, rank: 0 }]))
const started = performance.now()
for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const rotation = gameIndex % pool.length
  const strategies = pool.map((_, index) => pool[(index + rotation) % pool.length])
  const result = playArenaGame({
    playerCount: 5,
    deckMode: 'adaptive',
    strategies,
    seed: seed + gameIndex * 65537,
    championSamples: samples,
    legacySamples: Math.max(8, Math.round(samples / 2)),
  })
  const minimum = Math.min(...result.scores)
  const winners = result.scores.filter((score) => score === minimum).length
  result.strategies.forEach((id, player) => {
    const score = result.scores[player]
    const tied = result.scores.filter((value) => value === score).length
    stats[id].score += score
    stats[id].rank += 1 + result.scores.filter((value) => value < score).length + (tied - 1) / 2
    if (score === minimum) stats[id].wins += 1 / winners
  })
}
console.log(JSON.stringify({
  modelPath,
  games,
  strategy,
  seed,
  winRate: stats[strategy].wins / games,
  avgScore: stats[strategy].score / games,
  avgRank: stats[strategy].rank / games,
  opponents: Object.fromEntries(pool.slice(1).map((id) => [id, { winRate: stats[id].wins / games, avgScore: stats[id].score / games, avgRank: stats[id].rank / games }])),
  seconds: (performance.now() - started) / 1000,
}))
