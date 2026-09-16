import { createGame, resolveTurn } from '../game/engine.js'
import { chooseWebCard } from './ai-router.js'

function rngFor(seed) {
  let state = seed >>> 0
  return () => {
    state += 0x6d2b79f5
    let value = state
    value = Math.imul(value ^ value >>> 15, value | 1)
    value ^= value + Math.imul(value ^ value >>> 7, value | 61)
    return ((value ^ value >>> 14) >>> 0) / 4294967296
  }
}

export async function playWebArenaGame({ playerCount, deckMode, strategies, seed }, choose = chooseWebCard) {
  const deal = createGame(playerCount, deckMode, rngFor(seed))
  const hands = deal.hands.map(hand => [...hand])
  let rows = deal.rows, seenCards = [...deal.initialCards]
  const scores = Array(playerCount).fill(0), history = []
  for (let turn = 0; turn < 10; turn++) {
    const decisions = await Promise.all(hands.map((hand, seat) => {
      const order = Array.from({ length: playerCount }, (_, i) => (seat + i) % playerCount)
      return choose({ hand: [...hand], rows: rows.map(row => [...row]), seenCards: [...seenCards], deckSize: deal.deckSize,
        playerCount, scores: order.map(i => scores[i]), samples: 18, seed: seed + turn * 104729 + seat * 8191,
        playedCards: order.map(i => history.map(round => round[i])),
      }, strategies[seat])
    }))
    const cards = decisions.map((decision, seat) => {
      if (!hands[seat].includes(decision.card)) throw new Error('AI 返回了非法出牌')
      return decision.card
    })
    const result = resolveTurn({ rows, actions: cards.map((card, playerId) => ({ card, playerId })), humanPlayerId: -1 })
    rows = result.rows
    cards.forEach((card, i) => { hands[i].splice(hands[i].indexOf(card), 1); scores[i] += result.penalties[i] })
    seenCards.push(...cards)
    history.push(cards)
  }
  return { scores, strategies }
}
