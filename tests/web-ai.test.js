import test from 'node:test'
import assert from 'node:assert/strict'
import { modelFor, publicPayload, chooseWebCard } from '../src/web/ai-router.js'
import { playWebArenaGame } from '../src/web/arena-game.js'

const observation = { playerCount: 3, deckSize: 104, hand: [1, 2], rows: [[5], [10], [15], [20]], seenCards: [5, 10, 15, 20], scores: [0, 0, 0], seed: 123 }

test('strongest routes are specific to certified player counts and rules', () => {
  for (const p of [3, 4]) assert.equal(modelFor({ playerCount: p, deckSize: 104 }).id, 'distill2048-specialist2048')
  assert.equal(modelFor({ playerCount: 2, deckSize: 104 }).id, 'v6')
  for (const p of [2, 3, 4, 5, 10]) assert.equal(modelFor({ playerCount: p, deckSize: p * 10 + 4 }).id, 'neural_hybrid')
  assert.equal(modelFor({ playerCount: 5, deckSize: 104 }).certified, false)
})

test('HTTP payload excludes all other hands, identity hints and unrevealed choices', () => {
  const extra = { ...observation, players: [{ hand: [80] }], opponentHands: [[80]], opponentStrategies: ['random'], selectedCard: 80, history: [{ hidden: true }] }
  assert.deepEqual(publicPayload(extra), observation)
  const copy = publicPayload(extra)
  copy.hand.pop()
  copy.rows[0].push(7)
  assert.equal(observation.hand.length, 2)
  assert.deepEqual(observation.rows[0], [5])
})

test('champion response must identify the actual model and full budget', async () => {
  const good = { model: 'distill2048-specialist2048', worlds: 2048, card: 1 }
  const mock = value => async () => ({ ok: true, json: async () => value })
  assert.equal((await chooseWebCard(observation, 'strongest', { fetch: mock(good) })).card, 1)
  for (const bad of [{ ...good, card: 99 }, { ...good, worlds: 128 }, { ...good, model: 'legacy' }]) {
    await assert.rejects(chooseWebCard(observation, 'strongest', { fetch: mock(bad) }), /不一致/)
  }
})

test('service failure is reported instead of silently playing a random card', async () => {
  await assert.rejects(chooseWebCard(observation, 'strongest', { fetch: async () => { throw new TypeError('offline') } }), /无法连接/)
  await assert.rejects(chooseWebCard(observation, 'strongest', { fetch: async () => ({ ok: false, json: async () => ({ error: 'unavailable' }) }) }), /unavailable/)
})

test('arena completes ten simultaneous turns with isolated player observations', async () => {
  const observations = []
  const choose = async obs => {
    observations.push(structuredClone(obs))
    assert.ok(!('players' in obs) && !('opponentHands' in obs) && !('opponentStrategies' in obs))
    assert.equal(obs.seenCards.length, 4 + 3 * (10 - obs.hand.length))
    return { card: obs.hand[0] }
  }
  const setup = { playerCount: 3, deckMode: 'classic', strategies: ['a', 'b', 'c'], seed: 81234 }
  const result = await playWebArenaGame(setup, choose)
  assert.equal(observations.length, 30)
  assert.deepEqual(result, await playWebArenaGame(setup, async obs => ({ card: obs.hand[0] })))
  for (let round = 0; round < 10; round++) {
    const batch = observations.slice(round * 3, round * 3 + 3)
    assert.equal(new Set(batch.flatMap(obs => obs.hand)).size, (10 - round) * 3)
    assert.deepEqual(batch[1].scores, [...batch[0].scores.slice(1), batch[0].scores[0]])
  }
})
