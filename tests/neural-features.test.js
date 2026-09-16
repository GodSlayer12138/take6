import test from 'node:test'
import assert from 'node:assert/strict'
import {
  chooseNeuralCard,
  encodeNeuralCandidate,
  NEURAL_FEATURE_SIZE,
  NEURAL_FEATURE_VERSION,
  NEURAL_MODEL_FORMAT,
} from '../src/game/neural-features.js'

const observation = {
  rows: [[7, 9], [18], [31, 35, 38], [55]],
  hand: [3, 13, 19, 24, 29, 37, 41, 45, 51, 53],
  seenCards: [7, 9, 18, 31, 35, 38, 55],
  scores: [4, 7, 2, 11, 5],
  deckSize: 54,
  playerCount: 5,
}

test('神经特征固定为 270 维且不会编码隐藏手牌', () => {
  const features = encodeNeuralCandidate(observation, 24)
  assert.equal(features.length, NEURAL_FEATURE_SIZE)
  assert.ok(features.every((value) => value >= 0 && value <= 1))
  assert.equal(features[23], 1)
  assert.equal(features[99], 0)
})

test('浏览器神经运行时只会在合法手牌中选择', () => {
  const weight = Array(NEURAL_FEATURE_SIZE).fill(0)
  weight[252] = -1
  const model = {
    format: NEURAL_MODEL_FORMAT,
    featureVersion: NEURAL_FEATURE_VERSION,
    featureSize: NEURAL_FEATURE_SIZE,
    targetScale: 10,
    activation: 'relu',
    layers: [{ weight: [weight], bias: [0] }],
  }
  assert.equal(chooseNeuralCard(observation, model).card, 53)
})
