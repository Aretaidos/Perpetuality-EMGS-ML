import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Header from './components/layout/Header'
import ParticleBackground from './components/layout/ParticleBackground'
import HandVisualization from './components/hand/HandVisualization'
import EMGWaveform from './components/charts/EMGWaveform'
import MetricsPanel from './components/charts/MetricsPanel'
import GestureLog from './components/log/GestureLog'
import RecordingBrowser from './components/controls/RecordingBrowser'
import PlaybackControls from './components/controls/PlaybackControls'
import PredictionComparison from './components/charts/PredictionComparison'
import SessionMetrics from './components/charts/SessionMetrics'
import { Folder, Play, FileAudio } from 'lucide-react'
import {
  GestureType,
  GestureEvent,
  EMGDataPoint,
  ThumbAction
} from './types/gestures'

interface Recording {
  path: string
  filename: string
  duration_seconds: number
  num_samples: number
  num_gestures: number
  gesture_counts: Record<string, number>
}

interface GroundTruth {
  time: number
  gesture: string
  gesture_id: number
}

interface SessionMetricsData {
  total_predictions: number
  correct_predictions: number
  accuracy: number
  avg_latency_ms: number
  avg_confidence: number
  gesture_accuracies: Record<string, number>
}

// Map gesture names to finger states
function mapGestureToFingerState(gestureId: number | null): {
  indexPressed: boolean
  middlePressed: boolean
  thumbAction: ThumbAction
} {
  const state = {
    indexPressed: false,
    middlePressed: false,
    thumbAction: 'idle' as ThumbAction
  }

  if (gestureId === null) return state

  switch (gestureId) {
    case 0: // index_press
      state.indexPressed = true
      break
    case 2: // middle_press
      state.middlePressed = true
      break
    case 4: // thumb_click
      state.thumbAction = 'click'
      break
    case 5: // thumb_down
      state.thumbAction = 'down'
      break
    case 6: // thumb_in
      state.thumbAction = 'in'
      break
    case 7: // thumb_out
      state.thumbAction = 'out'
      break
    case 8: // thumb_up
      state.thumbAction = 'up'
      break
  }

  return state
}

function App() {
  // Recording state
  const [selectedRecording, setSelectedRecording] = useState<Recording | null>(null)
  const [browserOpen, setBrowserOpen] = useState(false)

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [totalDuration, setTotalDuration] = useState(0)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const [isLoaded, setIsLoaded] = useState(false)
  const [isComplete, setIsComplete] = useState(false)

  // Prediction state
  const [predictions, setPredictions] = useState<number[]>(Array(9).fill(0.1))
  const [detectedGesture, setDetectedGesture] = useState<number | null>(null)
  const [groundTruth, setGroundTruth] = useState<GroundTruth | null>(null)
  const [allGroundTruth, setAllGroundTruth] = useState<GroundTruth[]>([])
  const [confidence, setConfidence] = useState(0)
  const [isCorrect, setIsCorrect] = useState(false)

  // Visualization state
  const [emgBuffer, setEmgBuffer] = useState<EMGDataPoint[]>([])
  const [gestureLog, setGestureLog] = useState<GestureEvent[]>([])
  const [fingerState, setFingerState] = useState({
    indexPressed: false,
    middlePressed: false,
    thumbAction: 'idle' as ThumbAction
  })

  // Metrics state
  const [sessionMetrics, setSessionMetrics] = useState<SessionMetricsData | null>(null)
  const [liveMetrics, setLiveMetrics] = useState({
    latency: 0,
    accuracy: 0,
    modelConfidence: 0,
    samplesProcessed: 0,
  })

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'processing'>('disconnected')

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    const ws = new WebSocket(`ws://${window.location.hostname}:8000/ws/playback`)

    ws.onopen = () => {
      setConnectionStatus('connected')
      console.log('WebSocket connected')
    }

    ws.onclose = () => {
      setConnectionStatus('disconnected')
      console.log('WebSocket disconnected')
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setConnectionStatus('disconnected')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'loaded':
          setIsLoaded(true)
          setTotalDuration(data.duration)
          setAllGroundTruth(data.ground_truth || [])
          setIsComplete(false)
          setSessionMetrics(null)
          break

        case 'status':
          setIsPlaying(data.playing)
          setConnectionStatus(data.playing ? 'processing' : 'connected')
          break

        case 'frame':
          // Update all state from frame data
          setProgress(data.progress)
          setCurrentTime(data.timestamp)
          setPredictions(data.predictions)
          setDetectedGesture(data.detected_gesture)
          setGroundTruth(data.ground_truth)
          setConfidence(data.confidence)
          setIsCorrect(data.correct)

          // Update finger state based on detected gesture
          const newFingerState = mapGestureToFingerState(data.detected_gesture)
          setFingerState(newFingerState)

          // Update EMG buffer
          if (data.emg_data) {
            const newDataPoint: EMGDataPoint = {
              timestamp: Date.now(),
              channels: data.emg_data.map((ch: number[]) => ch[0] || 0)
            }
            setEmgBuffer(prev => [...prev.slice(-199), newDataPoint])
          }

          // Add to gesture log if detected
          if (data.detected_gesture !== null) {
            const event: GestureEvent = {
              id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              type: data.detected_gesture as GestureType,
              timestamp: Date.now(),
              confidence: data.confidence,
            }
            setGestureLog(prev => [event, ...prev].slice(0, 50))
          }

          // Update live metrics
          setLiveMetrics(prev => ({
            latency: data.latency_ms,
            modelConfidence: data.confidence,
            accuracy: prev.accuracy,
            samplesProcessed: prev.samplesProcessed + 1,
          }))
          break

        case 'complete':
          setIsPlaying(false)
          setIsComplete(true)
          setSessionMetrics(data.metrics)
          setConnectionStatus('connected')
          break

        case 'error':
          console.error('Server error:', data.message)
          break

        case 'seeked':
          setCurrentTime(data.time)
          setProgress(data.position / (selectedRecording?.num_samples || 1))
          break

        case 'speed_changed':
          setPlaybackSpeed(data.speed)
          break
      }
    }

    wsRef.current = ws
  }, [selectedRecording])

  // Load recording
  const loadRecording = useCallback((recording: Recording) => {
    setSelectedRecording(recording)
    setBrowserOpen(false)

    // Reset state
    setProgress(0)
    setCurrentTime(0)
    setTotalDuration(recording.duration_seconds)
    setIsLoaded(false)
    setIsComplete(false)
    setSessionMetrics(null)
    setEmgBuffer([])
    setGestureLog([])
    setPredictions(Array(9).fill(0.1))
    setDetectedGesture(null)
    setGroundTruth(null)

    // Connect and load
    connectWebSocket()

    // Wait for connection then send load command
    const checkAndLoad = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          action: 'load',
          file_path: recording.path
        }))
      } else {
        setTimeout(checkAndLoad, 100)
      }
    }
    checkAndLoad()
  }, [connectWebSocket])

  // Playback controls
  const handlePlay = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'play' }))
  }, [])

  const handlePause = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'pause' }))
  }, [])

  const handleSeek = useCallback((position: number) => {
    wsRef.current?.send(JSON.stringify({ action: 'seek', position }))
  }, [])

  const handleSpeedChange = useCallback((speed: number) => {
    wsRef.current?.send(JSON.stringify({ action: 'speed', value: speed }))
  }, [])

  const handleStep = useCallback(() => {
    wsRef.current?.send(JSON.stringify({ action: 'step' }))
  }, [])

  const handleReset = useCallback(() => {
    handleSeek(0)
    setGestureLog([])
    setSessionMetrics(null)
    setIsComplete(false)
  }, [handleSeek])

  // Connect on mount
  useEffect(() => {
    connectWebSocket()
    return () => {
      wsRef.current?.close()
    }
  }, [connectWebSocket])

  // Reset thumb action after delay
  useEffect(() => {
    if (fingerState.thumbAction !== 'idle') {
      const timer = setTimeout(() => {
        setFingerState(prev => ({ ...prev, thumbAction: 'idle' }))
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [fingerState.thumbAction])

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-gradient-to-br from-slate-50 to-indigo-50/30">
      <ParticleBackground />
      <div className="fixed inset-0 grid-pattern pointer-events-none" />

      <Header
        connectionStatus={connectionStatus}
        samplesProcessed={liveMetrics.samplesProcessed}
      />

      <main className="flex-1 container mx-auto px-4 py-6 relative z-10">
        {/* Recording selector bar */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-panel rounded-2xl p-4 mb-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <motion.button
                onClick={() => setBrowserOpen(true)}
                className="flex items-center gap-2 px-4 py-2.5 bg-gradient-to-r from-accent-primary to-accent-secondary text-white rounded-xl font-medium hover:opacity-90 transition-opacity"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Folder className="w-4 h-4" />
                Select Recording
              </motion.button>

              {selectedRecording && (
                <div className="flex items-center gap-3 px-4 py-2 bg-white/50 rounded-xl">
                  <FileAudio className="w-4 h-4 text-accent-primary" />
                  <div>
                    <p className="font-medium text-gray-800 text-sm">
                      {selectedRecording.filename}
                    </p>
                    <p className="text-xs text-gray-500">
                      {selectedRecording.num_gestures} gestures • {selectedRecording.duration_seconds.toFixed(1)}s
                    </p>
                  </div>
                </div>
              )}
            </div>

            {selectedRecording && isLoaded && (
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  {allGroundTruth.length} ground truth events
                </span>
                {!isPlaying && !isComplete && (
                  <motion.button
                    onClick={handlePlay}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-500 text-white rounded-xl font-medium hover:bg-emerald-600 transition-colors"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Play className="w-4 h-4" />
                    Start Testing
                  </motion.button>
                )}
              </div>
            )}
          </div>
        </motion.div>

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Hand visualization */}
          <motion.div
            className="lg:col-span-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="glass-panel rounded-2xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-800">Neural Interface</h2>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${
                    connectionStatus === 'connected' ? 'bg-status-connected' :
                    connectionStatus === 'processing' ? 'bg-status-processing animate-pulse' :
                    'bg-status-disconnected'
                  }`} />
                  <span className="text-sm text-gray-500 capitalize">{connectionStatus}</span>
                </div>
              </div>

              <HandVisualization
                indexPressed={fingerState.indexPressed}
                middlePressed={fingerState.middlePressed}
                thumbAction={fingerState.thumbAction}
                indexConfidence={fingerState.indexPressed ? confidence : 0}
                middleConfidence={fingerState.middlePressed ? confidence : 0}
                thumbConfidence={fingerState.thumbAction !== 'idle' ? confidence : 0}
              />
            </div>
          </motion.div>

          {/* Right column - Prediction comparison */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <PredictionComparison
              predictions={predictions}
              detectedGesture={detectedGesture}
              groundTruth={groundTruth}
              confidence={confidence}
              isCorrect={isCorrect}
            />
          </motion.div>
        </div>

        {/* Playback controls */}
        {selectedRecording && isLoaded && (
          <motion.div
            className="mt-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.15 }}
          >
            <PlaybackControls
              isPlaying={isPlaying}
              progress={progress}
              currentTime={currentTime}
              totalDuration={totalDuration}
              playbackSpeed={playbackSpeed}
              onPlay={handlePlay}
              onPause={handlePause}
              onSeek={handleSeek}
              onSpeedChange={handleSpeedChange}
              onStep={handleStep}
              onReset={handleReset}
              disabled={!isLoaded}
            />
          </motion.div>
        )}

        {/* EMG Waveform */}
        <motion.div
          className="mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <div className="glass-panel rounded-2xl p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">EMG Signal Monitor</h2>
            <EMGWaveform data={emgBuffer} />
          </div>
        </motion.div>

        {/* Bottom row - Metrics and Log */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Session Metrics */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.25 }}
          >
            <SessionMetrics
              metrics={sessionMetrics}
              isComplete={isComplete}
            />
          </motion.div>

          {/* Gesture Log */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <GestureLog events={gestureLog} />
          </motion.div>
        </div>

        {/* Live Metrics */}
        <motion.div
          className="mt-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.35 }}
        >
          <MetricsPanel metrics={liveMetrics} />
        </motion.div>
      </main>

      {/* Recording Browser Modal */}
      <RecordingBrowser
        isOpen={browserOpen}
        onClose={() => setBrowserOpen(false)}
        onSelect={loadRecording}
        selectedPath={selectedRecording?.path || null}
      />

      {/* No recording selected hint */}
      <AnimatePresence>
        {!selectedRecording && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 flex items-center justify-center pointer-events-none z-20"
          >
            <div className="glass-panel rounded-2xl p-8 text-center pointer-events-auto">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center">
                <FileAudio className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">
                Offline Testing Tool
              </h3>
              <p className="text-gray-500 mb-4 max-w-sm">
                Load an HDF5 recording to test your TinyML model with real EMG data and see predictions compared to ground truth.
              </p>
              <motion.button
                onClick={() => setBrowserOpen(true)}
                className="flex items-center gap-2 px-6 py-3 mx-auto bg-gradient-to-r from-accent-primary to-accent-secondary text-white rounded-xl font-medium hover:opacity-90 transition-opacity"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Folder className="w-5 h-5" />
                Browse Recordings
              </motion.button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default App
