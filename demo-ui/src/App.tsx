import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Header from './components/layout/Header'
import Footer from './components/layout/Footer'
import HandVisualization from './components/hand/HandVisualization'
import EMGWaveform from './components/charts/EMGWaveform'
import MetricsPanel from './components/charts/MetricsPanel'
import GestureLog from './components/log/GestureLog'
import ParticleBackground from './components/layout/ParticleBackground'
import {
  GestureState,
  GestureType,
  GestureEvent,
  EMGDataPoint,
  ThumbAction
} from './types/gestures'

// Initial state
const initialState: GestureState = {
  index: { pressed: false, confidence: 0, lastUpdate: 0 },
  middle: { pressed: false, confidence: 0, lastUpdate: 0 },
  thumb: { action: 'idle', confidence: 0, lastUpdate: 0 },
  emgBuffer: [],
  gestureLog: [],
  metrics: {
    latency: 0,
    accuracy: 0,
    modelConfidence: 0,
    samplesProcessed: 0,
  },
  connectionStatus: 'disconnected',
  isDemo: true,
}

// Generate demo EMG data
function generateDemoEMG(): number[] {
  return Array(7).fill(0).map(() => (Math.random() - 0.5) * 100 + Math.sin(Date.now() / 100) * 50)
}

function App() {
  const [state, setState] = useState<GestureState>(initialState)
  const [demoRunning, setDemoRunning] = useState(false)

  // Add gesture to log
  const addGestureEvent = useCallback((type: GestureType, confidence: number) => {
    const event: GestureEvent = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      timestamp: Date.now(),
      confidence,
    }

    setState(prev => ({
      ...prev,
      gestureLog: [event, ...prev.gestureLog].slice(0, 50), // Keep last 50 events
    }))
  }, [])

  // Handle gesture detection
  const handleGesture = useCallback((type: GestureType, confidence: number) => {
    const now = Date.now()

    setState(prev => {
      const newState = { ...prev }

      switch (type) {
        case GestureType.INDEX_PRESS:
          newState.index = { pressed: true, confidence, lastUpdate: now }
          break
        case GestureType.INDEX_RELEASE:
          newState.index = { pressed: false, confidence, lastUpdate: now }
          break
        case GestureType.MIDDLE_PRESS:
          newState.middle = { pressed: true, confidence, lastUpdate: now }
          break
        case GestureType.MIDDLE_RELEASE:
          newState.middle = { pressed: false, confidence, lastUpdate: now }
          break
        case GestureType.THUMB_CLICK:
          newState.thumb = { action: 'click', confidence, lastUpdate: now }
          setTimeout(() => {
            setState(s => ({ ...s, thumb: { ...s.thumb, action: 'idle' } }))
          }, 400)
          break
        case GestureType.THUMB_UP:
          newState.thumb = { action: 'up', confidence, lastUpdate: now }
          break
        case GestureType.THUMB_DOWN:
          newState.thumb = { action: 'down', confidence, lastUpdate: now }
          break
        case GestureType.THUMB_IN:
          newState.thumb = { action: 'in', confidence, lastUpdate: now }
          break
        case GestureType.THUMB_OUT:
          newState.thumb = { action: 'out', confidence, lastUpdate: now }
          break
      }

      return newState
    })

    addGestureEvent(type, confidence)
  }, [addGestureEvent])

  // Demo mode simulation
  useEffect(() => {
    if (!demoRunning) return

    // EMG data simulation
    const emgInterval = setInterval(() => {
      const newDataPoint: EMGDataPoint = {
        timestamp: Date.now(),
        channels: generateDemoEMG(),
      }

      setState(prev => ({
        ...prev,
        emgBuffer: [...prev.emgBuffer.slice(-199), newDataPoint],
        metrics: {
          ...prev.metrics,
          samplesProcessed: prev.metrics.samplesProcessed + 1,
          latency: 3.5 + Math.random() * 2,
          modelConfidence: 92 + Math.random() * 6,
          accuracy: 94 + Math.random() * 4,
        },
        connectionStatus: 'processing',
      }))
    }, 50) // 20Hz for demo visualization

    // Random gesture simulation
    const gestureInterval = setInterval(() => {
      const gestures = [
        GestureType.INDEX_PRESS,
        GestureType.INDEX_RELEASE,
        GestureType.MIDDLE_PRESS,
        GestureType.MIDDLE_RELEASE,
        GestureType.THUMB_CLICK,
        GestureType.THUMB_UP,
        GestureType.THUMB_DOWN,
      ]

      if (Math.random() > 0.7) {
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)]
        const confidence = 85 + Math.random() * 14
        handleGesture(randomGesture, confidence)
      }
    }, 800)

    return () => {
      clearInterval(emgInterval)
      clearInterval(gestureInterval)
    }
  }, [demoRunning, handleGesture])

  // Keyboard controls for manual testing
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const confidence = 95 + Math.random() * 4

      switch (e.key.toLowerCase()) {
        case 'i':
          handleGesture(state.index.pressed ? GestureType.INDEX_RELEASE : GestureType.INDEX_PRESS, confidence)
          break
        case 'm':
          handleGesture(state.middle.pressed ? GestureType.MIDDLE_RELEASE : GestureType.MIDDLE_PRESS, confidence)
          break
        case 't':
          handleGesture(GestureType.THUMB_CLICK, confidence)
          break
        case 'arrowup':
          handleGesture(GestureType.THUMB_UP, confidence)
          break
        case 'arrowdown':
          handleGesture(GestureType.THUMB_DOWN, confidence)
          break
        case 'arrowleft':
          handleGesture(GestureType.THUMB_IN, confidence)
          break
        case 'arrowright':
          handleGesture(GestureType.THUMB_OUT, confidence)
          break
        case ' ':
          e.preventDefault()
          setDemoRunning(prev => !prev)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleGesture, state.index.pressed, state.middle.pressed])

  // Update connection status
  useEffect(() => {
    setState(prev => ({
      ...prev,
      connectionStatus: demoRunning ? 'connected' : 'disconnected',
    }))
  }, [demoRunning])

  // Clear thumb direction after delay
  useEffect(() => {
    if (state.thumb.action !== 'idle' && state.thumb.action !== 'click') {
      const timer = setTimeout(() => {
        setState(prev => ({
          ...prev,
          thumb: { ...prev.thumb, action: 'idle' },
        }))
      }, 600)
      return () => clearTimeout(timer)
    }
  }, [state.thumb.action, state.thumb.lastUpdate])

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden">
      {/* Particle Background */}
      <ParticleBackground />

      {/* Grid Pattern Overlay */}
      <div className="fixed inset-0 grid-pattern pointer-events-none" />

      {/* Header */}
      <Header
        connectionStatus={state.connectionStatus}
        samplesProcessed={state.metrics.samplesProcessed}
      />

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-4 py-6 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Hand Visualization - Takes 2 columns */}
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="glass-panel rounded-2xl p-6 h-full">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Neural Interface</h2>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    state.connectionStatus === 'connected' ? 'bg-status-connected status-pulse' :
                    state.connectionStatus === 'processing' ? 'bg-status-processing status-pulse' :
                    'bg-status-disconnected'
                  }`} />
                  <span className="text-sm text-gray-500 capitalize">{state.connectionStatus}</span>
                </div>
              </div>

              <HandVisualization
                indexPressed={state.index.pressed}
                middlePressed={state.middle.pressed}
                thumbAction={state.thumb.action}
                indexConfidence={state.index.confidence}
                middleConfidence={state.middle.confidence}
                thumbConfidence={state.thumb.confidence}
              />
            </div>
          </motion.div>

          {/* Gesture Log - Side Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <GestureLog events={state.gestureLog} />
          </motion.div>
        </div>

        {/* EMG Waveform */}
        <motion.div
          className="mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="glass-panel rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">EMG Signal Monitor</h2>
            <EMGWaveform data={state.emgBuffer} />
          </div>
        </motion.div>

        {/* Metrics */}
        <motion.div
          className="mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <MetricsPanel metrics={state.metrics} />
        </motion.div>
      </main>

      {/* Footer */}
      <Footer
        demoRunning={demoRunning}
        onToggleDemo={() => setDemoRunning(prev => !prev)}
        onReset={() => {
          setState(initialState)
          setDemoRunning(false)
        }}
      />

      {/* Keyboard Help Tooltip */}
      <AnimatePresence>
        {!demoRunning && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="fixed bottom-24 left-1/2 transform -translate-x-1/2 glass-panel rounded-lg px-4 py-2 text-sm text-gray-600"
          >
            Press <kbd className="px-2 py-1 bg-gray-100 rounded text-xs font-mono mx-1">Space</kbd> to start demo,
            <kbd className="px-2 py-1 bg-gray-100 rounded text-xs font-mono mx-1">I</kbd> Index,
            <kbd className="px-2 py-1 bg-gray-100 rounded text-xs font-mono mx-1">M</kbd> Middle,
            <kbd className="px-2 py-1 bg-gray-100 rounded text-xs font-mono mx-1">T</kbd> Thumb
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
