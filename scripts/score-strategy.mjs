import { playArenaGame } from '../src/game/ai-core.js'

const strategy = process.argv[2] || 'neural_hybrid'
const games = Number(process.argv[3] || 100)
const seed = Number(process.argv[4] || 370000000)
const samples = Number(process.argv[5] || 18)
const deckMode = process.argv[6] === 'classic' ? 'classic' : 'adaptive'
const opponents = ['champion', 'external_mcs', 'cautious', 'random']
let wins = 0
let score = 0
let rank = 0
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  // Match benchmark-model.mjs exactly: rotate the strategy pool left while
  // keeping each deal seed paired with the same physical seat assignment.
  const focalSeat = (5 - (gameIndex % 5)) % 5
  const strategies = Array(5)
  strategies[focalSeat] = strategy
  opponents.forEach((opponent, offset) => {
    strategies[(focalSeat + offset + 1) % 5] = opponent
  })
  const result = playArenaGame({
    playerCount: 5,
    deckMode,
    strategies,
    seed: seed + gameIndex * 65537,
    championSamples: samples,
    legacySamples: Math.max(8, Math.round(samples / 2)),
  })
  const focalScore = result.scores[focalSeat]
  const minimum = Math.min(...result.scores)
  const tiedWinners = result.scores.filter((value) => value === minimum).length
  if (focalScore === minimum) wins += 1 / tiedWinners
  score += focalScore
  const lower = result.scores.filter((value) => value < focalScore).length
  const tied = result.scores.filter((value) => value === focalScore).length
  rank += 1 + lower + (tied - 1) / 2
}

console.log(JSON.stringify({
  strategy,
  games,
  seed,
  samples,
  deckMode,
  winRate: wins / games,
  avgScore: score / games,
  avgRank: rank / games,
  seconds: (performance.now() - started) / 1000,
}))
