/* eslint-env worker */
import { playWebArenaGame } from './arena-game.js'
import { POOLS, strategiesFor } from './strategy-catalog.js'

let STRATEGIES

function emptyStats(ids) {
  return Object.fromEntries(ids.map((id) => [id, { id, games: 0, wins: 0, scoreSum: 0, rankSum: 0, elo: 1500, cleanGames: 0 }]))
}

function snapshot(stats, headToHead) {
  return {
    strategies: Object.values(stats).map((entry) => ({
      ...entry,
      ...STRATEGIES[entry.id],
      winRate: entry.games ? entry.wins / entry.games : 0,
      avgScore: entry.games ? entry.scoreSum / entry.games : 0,
      avgRank: entry.games ? entry.rankSum / entry.games : 0,
      cleanRate: entry.games ? entry.cleanGames / entry.games : 0,
    })).sort((left, right) => right.elo - left.elo),
    headToHead,
  }
}

self.onmessage = async (event) => {
  try {
    const { games, playerCount, deckMode, seed = 20260820 } = event.data
    if (!POOLS[playerCount]) throw new Error('网页竞技场支持二至四人')
    const pool = POOLS[playerCount]
    STRATEGIES = strategiesFor({ playerCount, deckMode })
    const stats = emptyStats(pool)
    const headToHead = {
      champion: { gpu: 0, opponent: 0, ties: 0 },
      external: { gpu: 0, opponent: 0, ties: 0 },
    }

    for (let gameIndex = 0; gameIndex < games; gameIndex += 1) {
      const rotation = gameIndex % pool.length
      const seats = pool.map((_, index) => pool[(index + rotation) % pool.length])
      const result = await playWebArenaGame({
        playerCount,
        deckMode,
        strategies: seats,
        seed: seed + gameIndex * 65537,
      })
      const minimum = Math.min(...result.scores)
      const winners = result.scores.filter((score) => score === minimum).length
      const beforeElo = Object.fromEntries(pool.map((id) => [id, stats[id].elo]))
      const eloDelta = Object.fromEntries(pool.map((id) => [id, 0]))

      result.strategies.forEach((id, player) => {
        const score = result.scores[player]
        const lower = result.scores.filter((value) => value < score).length
        const tied = result.scores.filter((value) => value === score).length
        stats[id].games += 1
        stats[id].scoreSum += score
        stats[id].rankSum += 1 + lower + (tied - 1) / 2
        if (score === minimum) stats[id].wins += 1 / winners
        if (score === 0) stats[id].cleanGames += 1
      })

      for (let left = 0; left < pool.length; left += 1) {
        for (let right = left + 1; right < pool.length; right += 1) {
          const leftId = result.strategies[left]
          const rightId = result.strategies[right]
          const expectedLeft = 1 / (1 + 10 ** ((beforeElo[rightId] - beforeElo[leftId]) / 400))
          const actualLeft = result.scores[left] < result.scores[right] ? 1 : result.scores[left] === result.scores[right] ? 0.5 : 0
          const change = (20 / (pool.length - 1)) * (actualLeft - expectedLeft)
          eloDelta[leftId] += change
          eloDelta[rightId] -= change
        }
      }
      pool.forEach((id) => { stats[id].elo += eloDelta[id] })

      const gpuSeat = result.strategies.indexOf('strongest')
      for (const [key, opponentId] of [['champion', 'champion'], ['external', 'external_mcs']]) {
        const opponentSeat = result.strategies.indexOf(opponentId)
        if (gpuSeat < 0 || opponentSeat < 0) continue
        if (result.scores[gpuSeat] < result.scores[opponentSeat]) headToHead[key].gpu += 1
        else if (result.scores[gpuSeat] > result.scores[opponentSeat]) headToHead[key].opponent += 1
        else headToHead[key].ties += 1
      }

      if ((gameIndex + 1) % Math.max(1, Math.floor(games / 40)) === 0 || gameIndex === games - 1) {
        self.postMessage({ type: gameIndex === games - 1 ? 'complete' : 'progress', completed: gameIndex + 1, total: games, ...snapshot(stats, headToHead) })
      }
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
