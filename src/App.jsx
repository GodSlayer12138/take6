import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  Bot,
  BarChart3,
  BrainCircuit,
  Check,
  ChevronRight,
  CircleHelp,
  Cpu,
  Crown,
  Gauge,
  Layers3,
  RotateCcw,
  Settings2,
  ShieldAlert,
  Sparkles,
  Swords,
  Trophy,
  Users,
  X,
} from 'lucide-react'
import Arena from './Arena.jsx'
import ModelStatus from './web/ModelStatus.jsx'
import { modelFor } from './web/ai-router.js'
import {
  bullHeads,
  createGame,
  deckSizeFor,
  resolveTurn,
  rowPenalty,
} from './game/engine.js'

const AI_LEVELS = {
  balanced: { label: '敏捷', samples: 64, hint: '快速推演' },
  strong: { label: '强力', samples: 150, hint: '150×全局采样' },
  extreme: { label: '极限', samples: 320, hint: '高精度搜索' },
}

const BOT_NAMES = ['铁蹄', '赤角', '夜牧', '铜铃', '荒原', '霜角', '风蹄', '牧星', '黑曜']

const sampleRows = [
  [3, 8, 14],
  [21, 33],
  [44, 55, 61, 68],
  [72, 89],
]

function randomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    return globalThis.crypto.getRandomValues(new Uint32Array(1))[0]
  }
  return Math.floor(Math.random() * 2 ** 32)
}

function askAi(observation) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./web/ai.worker.js', import.meta.url), { type: 'module' })
    const timeout = window.setTimeout(() => {
      worker.terminate()
      reject(new Error('AI 推演超时'))
    }, 65000)

    worker.onmessage = (event) => {
      window.clearTimeout(timeout)
      worker.terminate()
      if (event.data.ok) resolve(event.data)
      else reject(new Error(event.data.message || 'AI 推演失败'))
    }
    worker.onerror = (event) => {
      window.clearTimeout(timeout)
      worker.terminate()
      reject(new Error(event.message || 'AI Worker 启动失败'))
    }
    worker.postMessage(observation)
  })
}

function Card({ value, selected = false, playable = false, compact = false, ghost = false, motion = '', onClick }) {
  const heads = value ? bullHeads(value) : 0
  const tone = heads >= 5 ? 'danger' : heads >= 3 ? 'warning' : heads === 2 ? 'warm' : 'plain'
  const Tag = playable ? 'button' : 'div'

  return (
    <Tag
      className={`game-card ${compact ? 'compact' : ''} ${selected ? 'selected' : ''} ${ghost ? 'ghost' : ''} ${motion} tone-${tone}`}
      onClick={onClick}
      type={playable ? 'button' : undefined}
      aria-label={value ? `${value}号牌，${heads}个牛头` : '暗牌'}
      aria-pressed={playable ? selected : undefined}
    >
      {ghost ? (
        <span className="card-back-mark">王</span>
      ) : (
        <>
          <span className="card-corner">{value}</span>
          <span className="card-number">{value}</span>
          <span className="card-bulls"><span>♉</span>{heads > 1 && <b>×{heads}</b>}</span>
        </>
      )}
    </Tag>
  )
}

function PlayerChip({ player, active, thinking }) {
  return (
    <div className={`player-chip ${player.isHuman ? 'human' : ''} ${active ? 'active' : ''}`}>
      <div className="player-avatar">{player.isHuman ? '你' : player.name.slice(0, 1)}</div>
      <div className="player-meta">
        <div className="player-name">
          {player.name}
          {thinking && <span className="thinking-dots"><i /><i /><i /></span>}
        </div>
        <div className="player-cards">{player.hand.length} 张手牌</div>
      </div>
      <div className="player-score"><b>{player.score}</b><span>牛头</span></div>
    </div>
  )
}

function GameRow({ row, index, selectable, onSelect, activeEvent, settledEvent }) {
  const danger = row.length >= 4
  const isClearing = activeEvent?.rowIndex === index && activeEvent.captured?.length > 0
  const targetSlot = activeEvent?.rowIndex === index && activeEvent.reason !== 'append' ? 0 : row.length
  const justSettled = settledEvent?.rowIndex === index ? settledEvent.card : null
  return (
    <button
      className={`table-row ${danger ? 'danger-row' : ''} ${selectable ? 'selectable' : ''} ${isClearing ? 'row-clearing' : ''}`}
      type="button"
      onClick={() => selectable && onSelect(index)}
      disabled={!selectable}
    >
      <div className="row-label">
        <span>行{index + 1}</span>
      </div>
      <div className="row-cards">
        <span
          className="fly-target"
          id={`row-target-${index}`}
          style={{ '--target-slot': targetSlot }}
          aria-hidden="true"
        />
        {row.map((card) => <Card key={card} value={card} compact motion={card === justSettled ? 'just-landed' : ''} />)}
        {Array.from({ length: Math.max(0, 5 - row.length) }, (_, slot) => (
          <div className="empty-slot" key={`slot-${slot}`} />
        ))}
      </div>
      <div className="row-risk">
        <span>{rowPenalty(row)}</span>
        <small>牛头</small>
      </div>
      {selectable && <div className="choose-row-cta">收这一行 <ChevronRight size={15} /></div>}
    </button>
  )
}

function rowsAfterEvents(startRows, events, completedCount) {
  const rows = startRows.map((row) => [...row])
  events.slice(0, completedCount).forEach((event) => {
    if (event.reason === 'append') rows[event.rowIndex].push(event.card)
    else rows[event.rowIndex] = [event.card]
  })
  return rows
}

function FlyingCard({ event, sequence }) {
  const [motion, setMotion] = useState(null)

  useLayoutEffect(() => {
    if (!event) {
      setMotion(null)
      return undefined
    }
    const source = document.getElementById(`reveal-card-${event.playerId}`)
    const target = document.getElementById(`row-target-${event.rowIndex}`)
    if (!source || !target) {
      setMotion(null)
      return undefined
    }
    const sourceRect = source.getBoundingClientRect()
    const targetRect = target.getBoundingClientRect()
    setMotion({
      left: sourceRect.left,
      top: sourceRect.top,
      dx: targetRect.left - sourceRect.left,
      dy: targetRect.top - sourceRect.top,
    })
    return undefined
  }, [event, sequence])

  if (!event || !motion) return null
  return (
    <div
      className="ntw-flight"
      style={{
        left: motion.left,
        top: motion.top,
        '--ntw-dx': `${motion.dx}px`,
        '--ntw-dy': `${motion.dy}px`,
      }}
      aria-hidden="true"
    >
      <Card value={event.card} compact />
    </div>
  )
}

function SetupScreen({ config, setConfig, onStart, onArena }) {
  const deckSize = deckSizeFor(config.playerCount, config.deckMode)

  return (
    <main className="setup-shell">
      <header className="setup-nav">
        <div className="brand-mark"><span>♉</span><div><b>牛头王</b><small>BULLHEAD KING</small></div></div>
        <button className="arena-entry" type="button" onClick={onArena}><BarChart3 size={15} /> AI 竞技场 <ChevronRight size={14} /></button>
      </header>

      <section className="setup-grid">
        <div className="setup-copy">
          <div className="setup-notice"><Sparkles size={14} /> 本地 AI 对战 · 所有推演只使用公开信息</div>
        </div>

        <div className="setup-panel">
          <div className="panel-heading">
            <div><span>本地对战</span><h2>创建一局</h2></div>
            <Settings2 />
          </div>

          <div className="setting-block">
            <div className="setting-label"><span><Users size={17} /> 玩家人数</span><b>{config.playerCount} 人</b></div>
            <div className="stepper">
              <button type="button" onClick={() => setConfig({ ...config, playerCount: Math.max(2, config.playerCount - 1) })}>−</button>
              <div>
                {Array.from({ length: 9 }, (_, index) => index + 2).map((count) => (
                  <button
                    key={count}
                    className={count === config.playerCount ? 'active' : ''}
                    onClick={() => setConfig({ ...config, playerCount: count })}
                    type="button"
                  >{count}</button>
                ))}
              </div>
              <button type="button" onClick={() => setConfig({ ...config, playerCount: Math.min(10, config.playerCount + 1) })}>＋</button>
            </div>
          </div>

          <div className="setting-block">
            <div className="setting-label"><span><Layers3 size={17} /> 牌池规则</span><b>{deckSize} 张</b></div>
            <div className="deck-options">
              <button
                type="button"
                className={config.deckMode === 'adaptive' ? 'active' : ''}
                onClick={() => setConfig({ ...config, deckMode: 'adaptive' })}
              >
                <span className="radio-dot"><i /></span>
                <span><b>精准牌池</b><small>10 × {config.playerCount} + 4，所有牌都在场</small></span>
                <em>{config.playerCount * 10 + 4}</em>
              </button>
              <button
                type="button"
                className={config.deckMode === 'classic' ? 'active' : ''}
                onClick={() => setConfig({ ...config, deckMode: 'classic' })}
              >
                <span className="radio-dot"><i /></span>
                <span><b>经典牌池</b><small>固定 1—104，保留未发牌的不确定性</small></span>
                <em>104</em>
              </button>
            </div>
          </div>

          <ModelStatus config={config} />
          {!modelFor(config).certified && <div className="setting-block compact-setting">
            <div className="setting-label"><span><Gauge size={17} /> AI 算力</span><b>{AI_LEVELS[config.aiLevel].hint}</b></div>
            <div className="level-tabs">
              {Object.entries(AI_LEVELS).map(([key, level]) => (
                <button
                  key={key}
                  className={config.aiLevel === key ? 'active' : ''}
                  onClick={() => setConfig({ ...config, aiLevel: key })}
                  type="button"
                >{level.label}</button>
              ))}
            </div>
          </div>

          }
          <button className="start-button" type="button" onClick={onStart}>
            <span><Swords size={20} /> 入座开局</span><ChevronRight />
          </button>
        </div>

        <div className="table-preview" aria-hidden="true">
          <div className="preview-glow" />
          <div className="preview-top"><span>牌桌预览</span><small>第 7 / 10 回合</small></div>
          {sampleRows.map((row, index) => (
            <div className="preview-row" key={index}>
              <span>0{index + 1}</span>
              <div>{row.map((card) => <Card key={card} value={card} compact />)}</div>
            </div>
          ))}
          <div className="preview-hand">
            {[17, 29, 46, 73, 91].map((card) => <Card key={card} value={card} compact />)}
          </div>
        </div>
      </section>
      <footer className="setup-footer"><span>规则内核：6 nimmt!</span><span>网页与本机 GPU 协作 · 不上传对局</span></footer>
    </main>
  )
}

function RulesModal({ onClose }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="rules-modal" role="dialog" aria-modal="true" aria-label="游戏规则" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-title"><div><span>QUICK GUIDE</span><h2>三分钟学会牛头王</h2></div><button onClick={onClose} type="button"><X /></button></div>
        <div className="rule-list">
          <div><b>01</b><span><strong>同时出牌</strong><small>每人暗出一张，然后按数字从小到大依次结算。</small></span></div>
          <div><b>02</b><span><strong>最小正差</strong><small>牌接在尾牌比它小、且差值最小的那一行。</small></span></div>
          <div><b>03</b><span><strong>第六头牛</strong><small>如果成为一行第六张，收走前五张并获得上面的牛头分。</small></span></div>
          <div><b>04</b><span><strong>小过所有行</strong><small>若牌比四行尾牌都小，由出牌者任选一行收走。</small></span></div>
        </div>
        <div className="score-guide">
          {[['普通牌', '1'], ['尾数 5', '2'], ['整十', '3'], ['11 倍数', '5'], ['55', '7']].map(([label, score]) => (
            <div key={label}><span>{label}</span><b>♉ × {score}</b></div>
          ))}
        </div>
        <button className="modal-done" type="button" onClick={onClose}>明白，回到牌桌</button>
      </section>
    </div>
  )
}

function eventText(event, players) {
  const player = players.find((item) => item.id === event.playerId)
  if (event.reason === 'sixth-card') return `${player.name} 的 ${event.card} 成为第六张，收下 ${event.penalty} 个牛头`
  if (event.reason === 'too-low') return `${player.name} 的 ${event.card} 太小，主动收走第 ${event.rowIndex + 1} 行`
  return `${player.name} 将 ${event.card} 接入第 ${event.rowIndex + 1} 行`
}

function GameScreen({ game, setGame, onRestart, onArena }) {
  const [showRules, setShowRules] = useState(false)
  const [aiError, setAiError] = useState('')
  const [advice, setAdvice] = useState(null)
  const [adviceLoading, setAdviceLoading] = useState(false)
  const adviceVersion = useRef(0)
  const resolutionEvents = game.turnResult?.events || []
  const resolutionIndex = game.resolveIndex ?? -1
  const activeEvent = game.phase === 'resolving' && resolutionIndex >= 0
    ? resolutionEvents[resolutionIndex]
    : null
  const settledEvent = game.phase === 'resolving' && resolutionIndex > 0
    ? resolutionEvents[resolutionIndex - 1]
    : null
  const displayRows = game.phase === 'row-choice'
    ? game.pending.previewRows
    : game.phase === 'resolving'
      ? rowsAfterEvents(game.turnResult.startRows, resolutionEvents, Math.max(0, resolutionIndex))
      : game.rows
  const isChoosing = game.phase === 'choosing'
  const selected = game.selectedCard
  const human = game.players[0]
  const ranking = useMemo(
    () => [...game.players].sort((left, right) => left.score - right.score),
    [game.players],
  )

  useEffect(() => {
    if (game.phase !== 'resolving') return undefined
    const delay = resolutionIndex < 0 ? 620 : 680
    const timer = window.setTimeout(() => {
      setGame((current) => {
        if (!current || current.matchId !== game.matchId || current.phase !== 'resolving') return current
        const nextIndex = (current.resolveIndex ?? -1) + 1
        if (nextIndex >= current.turnResult.events.length) {
          return {
            ...current,
            phase: current.postResolvePhase,
            resolveIndex: null,
            postResolvePhase: null,
          }
        }
        return { ...current, resolveIndex: nextIndex }
      })
    }, delay)
    return () => window.clearTimeout(timer)
  }, [game.matchId, game.phase, resolutionIndex, setGame])

  useEffect(() => {
    if (game.phase !== 'turn-result') return undefined
    const timer = window.setTimeout(() => {
      adviceVersion.current += 1
      setAdvice(null)
      setAdviceLoading(false)
      setGame((current) => {
        if (!current || current.matchId !== game.matchId || current.phase !== 'turn-result') return current
        return {
          ...current,
          turn: current.turn + 1,
          phase: 'choosing',
          revealed: null,
          turnResult: null,
          resolveIndex: null,
          postResolvePhase: null,
        }
      })
    }, 900)
    return () => window.clearTimeout(timer)
  }, [game.matchId, game.phase, setGame])

  function observationFor(player, samples = AI_LEVELS[game.aiLevel].samples) {
    const playerIndex = game.players.findIndex((candidate) => candidate.id === player.id)
    const relativePlayers = Array.from({ length: game.playerCount }, (_, offset) => game.players[(playerIndex + offset) % game.playerCount])
    return {
      rows: game.rows,
      hand: player.hand,
      seenCards: game.seenCards,
      deckSize: game.deckSize,
      playerCount: game.playerCount,
      scores: relativePlayers.map((candidate) => candidate.score),
      playedCards: relativePlayers.map((candidate) => game.history.map((round) => round.find((event) => event.playerId === candidate.id)?.card).filter(Number.isInteger)),
      samples,
      seed: randomSeed(),
    }
  }

  function completeTurn(baseGame, actions, humanRowChoice = null) {
    const resolution = resolveTurn({
      rows: baseGame.rows,
      actions,
      humanPlayerId: 0,
      humanRowChoice,
    })

    if (resolution.needsChoice) {
      return {
        ...baseGame,
        phase: 'row-choice',
        revealed: actions,
        pending: { actions, previewRows: resolution.previewRows },
      }
    }

    const players = baseGame.players.map((player) => {
      const action = actions.find((item) => item.playerId === player.id)
      return {
        ...player,
        hand: player.hand.filter((card) => card !== action.card),
        score: player.score + resolution.penalties[player.id],
        captured: [...player.captured, ...resolution.capturedByPlayer[player.id]],
      }
    })
    const isLastTurn = baseGame.turn === 10

    const postResolvePhase = isLastTurn ? 'game-over' : 'turn-result'

    return {
      ...baseGame,
      players,
      rows: resolution.rows,
      seenCards: [...baseGame.seenCards, ...actions.map((action) => action.card)],
      selectedCard: null,
      phase: 'resolving',
      revealed: actions,
      pending: null,
      turnResult: {
        ...resolution,
        startRows: baseGame.rows.map((row) => [...row]),
      },
      resolveIndex: -1,
      postResolvePhase,
      history: [...baseGame.history, resolution.events],
    }
  }

  async function submitCard() {
    if (!isChoosing || selected == null) return
    const snapshot = game
    const matchId = game.matchId
    adviceVersion.current += 1
    setAdvice(null)
    setGame({ ...game, phase: 'thinking' })

    setAiError('')
    let decisions
    try {
      decisions = await Promise.all(snapshot.players.slice(1).map(player => askAi(observationFor(player))))
    } catch (error) {
      setGame(current => {
        if (!current || current.matchId !== matchId) return current
        return { ...current, phase: 'choosing' }
      })
      setAiError(error.message || 'AI 推演失败，请重试当前回合')
      return
    }
    const actions = [
      { playerId: 0, card: selected },
      ...decisions.map((decision, index) => ({ playerId: index + 1, card: decision.card })),
    ]

    setGame((current) => {
      if (!current || current.matchId !== matchId) return current
      return completeTurn(snapshot, actions)
    })
  }

  async function getAdvice() {
    if (!isChoosing || adviceLoading) return
    const version = ++adviceVersion.current
    setAdviceLoading(true)
    try {
      const result = await askAi(observationFor(human, Math.max(180, AI_LEVELS[game.aiLevel].samples)))
      if (version === adviceVersion.current) setAdvice(result)
    } catch (error) {
      if (version === adviceVersion.current) setAdvice({ error: error.message || '暂时无法分析' })
    } finally {
      if (version === adviceVersion.current) setAdviceLoading(false)
    }
  }

  function chooseRow(index) {
    setGame(completeTurn(game, game.pending.actions, index))
  }

  const phaseText = game.phase === 'row-choice'
    ? '你的牌比所有尾牌都小，请选择收走一行'
    : isChoosing
      ? '等待你选择手牌'
      : game.phase === 'thinking'
        ? '对手正在思考'
        : game.phase === 'resolving'
          ? resolutionIndex < 0 ? '所有人已出牌，开始翻牌' : `正在结算第 ${resolutionIndex + 1} / ${resolutionEvents.length} 张牌`
          : '本回合已结算'
  const latestRound = game.history.at(-1) || []
  const latestLog = latestRound.find((event) => event.penalty > 0) || latestRound.at(-1)

  return (
    <main className="game-shell">
      <header className="game-nav">
        <div className="brand-mark small"><span>♉</span><div><b>牛头王</b><small>BULLHEAD KING</small></div></div>
        <div className="round-simple">第 {game.turn} / 10 轮 · {game.deckSize} 张牌 · {modelFor(game).label}</div>
        <div className="nav-actions">
          <button type="button" onClick={onArena}><BarChart3 /> 竞技场</button>
          <button type="button" onClick={() => setShowRules(true)}><CircleHelp /> 规则</button>
          <button type="button" onClick={onRestart}><RotateCcw /> 重开</button>
        </div>
      </header>

      <section className="game-layout compact-game-layout">
        <section className="board-area compact-board">
          <div className="compact-status" role="status" aria-live="polite">
            <span>{game.deckMode === 'adaptive' ? '精准牌池' : '经典牌池'}（1—{game.deckSize}） · {game.playerCount} 人 · 牛头最少者获胜</span>
            <b>第 {game.turn} 轮，{phaseText}</b>
          </div>
          {aiError && <div className="ai-error" role="alert">{aiError}。本回合尚未出牌，可重试。</div>}
          <div className="rows-stack">
            {displayRows.map((row, index) => (
              <GameRow
                key={index}
                row={row}
                index={index}
                selectable={game.phase === 'row-choice'}
                onSelect={chooseRow}
                activeEvent={activeEvent}
                settledEvent={settledEvent}
              />
            ))}
          </div>

          <div className={`reveal-zone compact-reveal ${game.revealed ? 'has-cards' : ''} ${game.phase === 'resolving' ? 'is-resolving' : ''}`}>
            {game.phase === 'thinking' && <div className="compact-thinking"><Cpu /> 对手正在推演…</div>}
            {game.revealed && [...game.revealed].sort((a, b) => a.card - b.card).map((action, index) => {
              const player = game.players.find((item) => item.id === action.playerId)
              const alreadyPlaced = game.phase === 'resolving' && resolutionIndex > index
              return <div id={`reveal-card-${action.playerId}`} className={`revealed-card ${alreadyPlaced ? 'placed' : ''}`} style={{ '--reveal-order': index }} key={action.playerId}><Card value={action.card} compact /><span>{player.name}</span></div>
            })}
          </div>

          <section className="opponents compact-seats" style={{ '--player-count': game.playerCount }}>
            {game.players.map((player) => (
              <PlayerChip key={player.id} player={player} active={player.isHuman} thinking={game.phase === 'thinking' && !player.isHuman} />
            ))}
          </section>

          <div className={`compact-log ${latestLog?.penalty ? 'penalty-log' : ''}`}>
            {latestLog ? eventText(latestLog, game.players) : '所有人同时暗出一张牌，锁定后统一翻开。'}
          </div>
        </section>
      </section>

      <FlyingCard event={activeEvent} sequence={`${game.turn}-${resolutionIndex}`} />

      <section className="hand-dock simple-hand">
        <div className="hand-heading">
          <div><h3>我的手牌</h3><span>{human.hand.length} 张</span></div>
          <p>{isChoosing ? '选择一张牌打出' : game.phase === 'row-choice' ? '请先选择要收走的行' : '等待结算'}</p>
        </div>
        <div className="hand-cards">
          {human.hand.map((card) => (
            <Card
              key={card}
              value={card}
              selected={selected === card}
              playable={isChoosing}
              onClick={() => {
                adviceVersion.current += 1
                setGame({ ...game, selectedCard: card })
              }}
            />
          ))}
        </div>
        <div className="hand-actions">
          <div className="compact-coach">
            {!advice && !adviceLoading && <button type="button" onClick={getAdvice} disabled={!isChoosing}><BrainCircuit /> AI 建议</button>}
            {adviceLoading && <span><Cpu /> 分析中…</span>}
            {advice?.error && <button type="button" onClick={getAdvice} disabled={!isChoosing}>{advice.error} · 重试</button>}
            {advice?.evaluations && (
              <div className="compact-advice">
                <span>建议</span>
                {advice.evaluations.slice(0, 3).map((item, index) => (
                  <button key={item.card} type="button" onClick={() => setGame({ ...game, selectedCard: item.card })}>
                    {index === 0 ? '首选 ' : ''}{item.card}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button className="lock-button" type="button" disabled={!isChoosing || selected == null} onClick={submitCard}>
            {selected == null ? <span>请选择一张牌</span> : <span><Check /> 出 {selected}</span>}
          </button>
        </div>
      </section>

      {game.phase === 'turn-result' && (
        <div className="result-toast auto-next" role="status" aria-live="polite">
          <div className={game.turnResult.penalties[0] ? 'bad' : 'good'}>{game.turnResult.penalties[0] ? <ShieldAlert /> : <Check />}</div>
          <span><b>{game.turnResult.penalties[0] ? `本回合 +${game.turnResult.penalties[0]} 牛头` : '安全落牌'}</b><small>{game.turnResult.penalties[0] ? '风险已经结算，正在进入下一回合。' : '没有收走任何牌，正在进入下一回合。'}</small></span>
          <i className="auto-next-progress" aria-hidden="true" />
        </div>
      )}

      {game.phase === 'game-over' && (
        <div className="modal-backdrop end-backdrop">
          <section className="end-modal">
            <div className="winner-crown"><Crown /></div>
            <span>FINAL RESULT</span>
            <h2>{ranking[0].isHuman ? '你统治了这片牧场！' : `${ranking[0].name} 拿下本局`}</h2>
            <p>十回合结束，牛头最少者获胜。</p>
            <div className="ranking-list">
              {ranking.map((player, index) => (
                <div key={player.id} className={player.isHuman ? 'human-rank' : ''}>
                  <em>{index === 0 ? <Trophy size={18} /> : index + 1}</em>
                  <span>{player.name}<small>{player.captured.length} 张罚牌</small></span>
                  <b>{player.score}<small>牛头</small></b>
                </div>
              ))}
            </div>
            <button type="button" onClick={onRestart}><RotateCcw /> 再开一局</button>
          </section>
        </div>
      )}

      {showRules && <RulesModal onClose={() => setShowRules(false)} />}
    </main>
  )
}

export default function App() {
  const [config, setConfig] = useState({ playerCount: 4, deckMode: 'classic', aiLevel: 'strong' })
  const [game, setGame] = useState(null)
  const [showArena, setShowArena] = useState(false)

  function startGame() {
    const deal = createGame(config.playerCount, config.deckMode)
    const players = deal.hands.map((hand, index) => ({
      id: index,
      name: index === 0 ? '你' : BOT_NAMES[index - 1],
      isHuman: index === 0,
      hand,
      score: 0,
      captured: [],
    }))
    setGame({
      matchId: `${Date.now()}-${randomSeed()}`,
      playerCount: config.playerCount,
      deckMode: config.deckMode,
      aiLevel: config.aiLevel,
      deckSize: deal.deckSize,
      players,
      rows: deal.rows,
      seenCards: deal.initialCards,
      turn: 1,
      phase: 'choosing',
      selectedCard: null,
      revealed: null,
      pending: null,
      turnResult: null,
      resolveIndex: null,
      postResolvePhase: null,
      history: [],
    })
  }

  if (showArena) return <Arena onBack={() => setShowArena(false)} />
  if (!game) return <SetupScreen config={config} setConfig={setConfig} onStart={startGame} onArena={() => setShowArena(true)} />
  return <GameScreen game={game} setGame={setGame} onRestart={() => setGame(null)} onArena={() => setShowArena(true)} />
}
