import test from 'node:test'
import assert from 'node:assert/strict'
import {
  bullHeads,
  createGame,
  deckSizeFor,
  resolveTurn,
  targetRowIndex,
} from '../src/game/engine.js'

test('牛头分值遵循 55、11、10、5 的优先级', () => {
  assert.equal(bullHeads(55), 7)
  assert.equal(bullHeads(22), 5)
  assert.equal(bullHeads(40), 3)
  assert.equal(bullHeads(35), 2)
  assert.equal(bullHeads(37), 1)
})

test('牌进入尾牌小于它且差值最小的行', () => {
  assert.equal(targetRowIndex([[12], [37], [43], [58]], 45), 2)
  assert.equal(targetRowIndex([[12], [37], [43], [58]], 10), -1)
})

test('第六张牌收走原有五张牌', () => {
  const result = resolveTurn({
    rows: [[1, 2, 3, 4, 5], [20], [40], [60]],
    actions: [{ playerId: 1, card: 6 }],
  })
  assert.deepEqual(result.rows[0], [6])
  assert.equal(result.penalties[1], 6)
  assert.equal(result.events[0].reason, 'sixth-card')
})

test('人类打出最低牌时暂停并等待选行', () => {
  const pending = resolveTurn({
    rows: [[12], [37], [43], [58]],
    actions: [{ playerId: 0, card: 10 }],
  })
  assert.equal(pending.needsChoice, true)

  const resolved = resolveTurn({
    rows: [[12], [37], [43], [58]],
    actions: [{ playerId: 0, card: 10 }],
    humanRowChoice: 0,
  })
  assert.deepEqual(resolved.rows[0], [10])
  assert.equal(resolved.penalties[0], 1)
})

test('两种牌池大小正确且发牌不重复', () => {
  assert.equal(deckSizeFor(4, 'adaptive'), 44)
  assert.equal(deckSizeFor(4, 'classic'), 104)
  const game = createGame(4, 'adaptive', () => 0.42)
  const dealt = [...game.hands.flat(), ...game.rows.flat()]
  assert.equal(dealt.length, 44)
  assert.equal(new Set(dealt).size, 44)
})
