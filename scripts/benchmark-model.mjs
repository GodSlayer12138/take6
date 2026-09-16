import { playArenaGame, STRATEGIES } from '../src/game/ai-core.js'

const games = Number(process.argv[2] || 200)
const deckMode = process.argv[3] === 'classic' ? 'classic' : 'adaptive'
const pool = (process.argv[4] || 'neural_hybrid,champion,external_mcs,cautious,random').split(',')
const seed = Number(process.argv[5] || 62000000)
const championSamples = Number(process.argv[6] || (games >= 200 ? 18 : 24))
const playerCount = pool.length
if (playerCount < 2 || playerCount > 10) throw new Error('策略数量必须为 2—10')

const stats = Object.fromEntries(pool.map((id) => [id, { games: 0, wins: 0, score: 0, rank: 0 }]))
const headToHead = Object.fromEntries(pool.map((left) => [left, Object.fromEntries(pool.map((right) => [right, { wins: 0, losses: 0, ties: 0 }]))]))
const started = performance.now()

for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
  const rotation = gameIndex % pool.length
  const strategies = pool.map((_, index) => pool[(index + rotation) % pool.length])
  const result = playArenaGame({
    playerCount,
    deckMode,
    strategies,
    seed: seed + gameIndex * 65537,
    championSamples,
    legacySamples: Math.max(8, Math.round(championSamples / 2)),
  })
  const minimum = Math.min(...result.scores)
  const winners = result.scores.filter((score) => score === minimum).length
  result.strategies.forEach((id, player) => {
    const score = result.scores[player]
    const lower = result.scores.filter((value) => value < score).length
    const tied = result.scores.filter((value) => value === score).length
    stats[id].games += 1
    stats[id].score += score
    stats[id].rank += 1 + lower + (tied - 1) / 2
    if (score === minimum) stats[id].wins += 1 / winners
  })
  for (let left = 0; left < playerCount; left += 1) {
    for (let right = left + 1; right < playerCount; right += 1) {
      const leftId = result.strategies[left]
      const rightId = result.strategies[right]
      if (result.scores[left] < result.scores[right]) {
        headToHead[leftId][rightId].wins += 1
        headToHead[rightId][leftId].losses += 1
      } else if (result.scores[left] > result.scores[right]) {
        headToHead[leftId][rightId].losses += 1
        headToHead[rightId][leftId].wins += 1
      } else {
        headToHead[leftId][rightId].ties += 1
        headToHead[rightId][leftId].ties += 1
      }
    }
  }
}

const table = pool.map((id) => ({
  id,
  strategy: STRATEGIES[id]?.name || id,
  winRate: `${(stats[id].wins / games * 100).toFixed(1)}%`,
  avgScore: Number((stats[id].score / games).toFixed(3)),
  avgRank: Number((stats[id].rank / games).toFixed(3)),
}))
console.table(table)
if (process.argv[7] !== 'summary') console.log(JSON.stringify({ games, deckMode, seed, championSamples, table, headToHead }, null, 2))
console.log(`${games} ${deckMode} games in ${((performance.now() - started) / 1000).toFixed(2)}s`)
