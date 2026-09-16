import { playArenaGame, STRATEGIES } from '../src/game/ai-core.js'

const games = Number(process.argv[2] || 100)
const deckMode = process.argv[3] === 'classic' ? 'classic' : 'adaptive'
const playerCount = 5
const pool = ['neural_hybrid', 'champion', 'external_mcs', 'cautious', 'random']
const stats = Object.fromEntries(pool.map((id) => [id, { wins: 0, score: 0, rank: 0 }]))
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const rotation = gameIndex % pool.length
  const strategies = pool.map((_, index) => pool[(index + rotation) % pool.length])
  const result = playArenaGame({
    playerCount,
    deckMode,
    strategies,
    seed: 20260820 + gameIndex * 65537,
    championSamples: games > 50 ? 18 : 24,
    legacySamples: games > 50 ? 9 : 12,
  })
  const minimum = Math.min(...result.scores)
  const winners = result.scores.filter((score) => score === minimum).length
  result.strategies.forEach((id, player) => {
    const score = result.scores[player]
    const lower = result.scores.filter((value) => value < score).length
    const tied = result.scores.filter((value) => value === score).length
    stats[id].score += score
    stats[id].rank += 1 + lower + (tied - 1) / 2
    if (score === minimum) stats[id].wins += 1 / winners
  })
}

const table = pool.map((id) => ({
  strategy: STRATEGIES[id].name,
  winRate: `${(stats[id].wins / games * 100).toFixed(1)}%`,
  avgScore: (stats[id].score / games).toFixed(2),
  avgRank: (stats[id].rank / games).toFixed(2),
}))

console.table(table)
console.log(`${games} ${deckMode} games in ${((performance.now() - started) / 1000).toFixed(2)}s`)
