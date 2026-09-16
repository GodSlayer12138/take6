import test from 'node:test'
import assert from 'node:assert/strict'
import {
  chooseConvertedModelCard,
  chooseExternalMcsCard,
  encodeRl6NimmtState,
  normalizeRl6NimmtInput,
  RL6NIMMT_FORMAT,
} from '../src/game/external-adapter.js'

const observation = {
  rows: [[7], [18], [31], [39]],
  hand: [3, 13, 19, 24, 29, 37, 41, 45, 51, 53],
  seenCards: [7, 18, 31, 39],
  deckSize: 54,
  playerCount: 5,
  seed: 20260820,
}

test('适配器生成上游兼容的 47 维、零基牌号状态', () => {
  const state = encodeRl6NimmtState(observation)
  assert.equal(state.length, 47)
  assert.deepEqual(state.slice(0, 3), [2, 12, 18])
  assert.equal(state[10], 5)
  assert.equal(state[23], 6)
  assert.equal(state[24], -1)
  assert.equal(normalizeRl6NimmtInput(state).length, 47)
  assert.equal(normalizeRl6NimmtInput(state, { action: 3 }).length, 48)
})

test('转换后的 action-conditioned 策略只在合法手牌中选最大 logit', () => {
  const weight = Array(48).fill(0)
  weight[0] = 1
  const model = {
    format: RL6NIMMT_FORMAT,
    architecture: 'action-conditioned-policy',
    cards: 104,
    activation: 'relu',
    layers: [{ weight: [weight], bias: [0] }],
  }
  assert.equal(chooseConvertedModelCard(observation, model).card, 53)
})

test('公开 MCS 适配策略可复现且返回合法牌', () => {
  const first = chooseExternalMcsCard(observation, { mcMax: 20 })
  const second = chooseExternalMcsCard(observation, { mcMax: 20 })
  assert.equal(first.card, second.card)
  assert.ok(observation.hand.includes(first.card))
})
