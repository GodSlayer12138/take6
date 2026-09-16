export const HAND_SIZE = 10
export const ROW_COUNT = 4
export const ROW_LIMIT = 5

export function bullHeads(card) {
  if (card === 55) return 7
  if (card % 11 === 0) return 5
  if (card % 10 === 0) return 3
  if (card % 10 === 5) return 2
  return 1
}

export function rowPenalty(row) {
  return row.reduce((sum, card) => sum + bullHeads(card), 0)
}

export function deckSizeFor(playerCount, deckMode) {
  return deckMode === 'adaptive' ? playerCount * HAND_SIZE + ROW_COUNT : 104
}

export function makeDeck(size) {
  return Array.from({ length: size }, (_, index) => index + 1)
}

export function shuffle(items, rng = Math.random) {
  const result = [...items]
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(rng() * (index + 1))
    ;[result[index], result[swapIndex]] = [result[swapIndex], result[index]]
  }
  return result
}

export function createGame(playerCount, deckMode, rng = Math.random) {
  const deckSize = deckSizeFor(playerCount, deckMode)
  const deck = shuffle(makeDeck(deckSize), rng)
  const hands = Array.from({ length: playerCount }, () => [])

  for (let cardIndex = 0; cardIndex < HAND_SIZE; cardIndex += 1) {
    for (let playerIndex = 0; playerIndex < playerCount; playerIndex += 1) {
      hands[playerIndex].push(deck.pop())
    }
  }

  hands.forEach((hand) => hand.sort((a, b) => a - b))
  const rows = Array.from({ length: ROW_COUNT }, () => [deck.pop()])

  return {
    deckSize,
    hands,
    rows,
    initialCards: rows.flat(),
  }
}

export function targetRowIndex(rows, card) {
  let bestIndex = -1
  let bestTail = -Infinity
  rows.forEach((row, index) => {
    const tail = row[row.length - 1]
    if (tail < card && tail > bestTail) {
      bestTail = tail
      bestIndex = index
    }
  })
  return bestIndex
}

export function cheapestRowIndex(rows) {
  let bestIndex = 0
  let bestPenalty = Infinity
  rows.forEach((row, index) => {
    const penalty = rowPenalty(row)
    if (penalty < bestPenalty || (penalty === bestPenalty && row.length < rows[bestIndex].length)) {
      bestPenalty = penalty
      bestIndex = index
    }
  })
  return bestIndex
}

/**
 * Resolves a simultaneous turn. If the human plays below all row tails and has
 * not chosen a row yet, the function pauses with the exact public board at that
 * moment. Calling it again with humanRowChoice resolves deterministically.
 */
export function resolveTurn({ rows, actions, humanPlayerId = 0, humanRowChoice = null }) {
  const nextRows = rows.map((row) => [...row])
  const ordered = [...actions].sort((left, right) => left.card - right.card)
  const penalties = Object.fromEntries(actions.map((action) => [action.playerId, 0]))
  const capturedByPlayer = Object.fromEntries(actions.map((action) => [action.playerId, []]))
  const events = []

  for (const action of ordered) {
    let rowIndex = targetRowIndex(nextRows, action.card)
    let reason = 'append'

    if (rowIndex === -1) {
      if (action.playerId === humanPlayerId && humanRowChoice == null) {
        return {
          needsChoice: true,
          choiceFor: action,
          previewRows: nextRows,
          events,
        }
      }
      rowIndex = action.playerId === humanPlayerId ? humanRowChoice : cheapestRowIndex(nextRows)
      reason = 'too-low'
    }

    let captured = []
    if (reason === 'too-low' || nextRows[rowIndex].length >= ROW_LIMIT) {
      captured = [...nextRows[rowIndex]]
      nextRows[rowIndex] = [action.card]
      if (reason !== 'too-low') reason = 'sixth-card'
    } else {
      nextRows[rowIndex].push(action.card)
    }

    const penalty = rowPenalty(captured)
    penalties[action.playerId] += penalty
    capturedByPlayer[action.playerId].push(...captured)
    events.push({ ...action, rowIndex, reason, captured, penalty })
  }

  return {
    needsChoice: false,
    rows: nextRows,
    events,
    penalties,
    capturedByPlayer,
  }
}
