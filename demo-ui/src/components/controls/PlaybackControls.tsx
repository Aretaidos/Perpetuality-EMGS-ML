import { motion } from 'framer-motion'
import {
  Play, Pause, SkipForward, RotateCcw,
  Gauge, ChevronLeft, ChevronRight
} from 'lucide-react'

interface PlaybackControlsProps {
  isPlaying: boolean
  progress: number
  currentTime: number
  totalDuration: number
  playbackSpeed: number
  onPlay: () => void
  onPause: () => void
  onSeek: (position: number) => void
  onSpeedChange: (speed: number) => void
  onStep: () => void
  onReset: () => void
  disabled?: boolean
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 2, 4]

export default function PlaybackControls({
  isPlaying,
  progress,
  currentTime,
  totalDuration,
  playbackSpeed,
  onPlay,
  onPause,
  onSeek,
  onSpeedChange,
  onStep,
  onReset,
  disabled = false
}: PlaybackControlsProps) {
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 100)
    return `${mins}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  return (
    <div className="glass-panel rounded-2xl p-4">
      {/* Progress Bar */}
      <div className="mb-4">
        <div
          className="relative h-2 bg-gray-200 rounded-full cursor-pointer overflow-hidden"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const pos = (e.clientX - rect.left) / rect.width
            onSeek(Math.max(0, Math.min(1, pos)))
          }}
        >
          {/* Progress fill */}
          <motion.div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-accent-primary to-accent-secondary rounded-full"
            style={{ width: `${progress * 100}%` }}
            transition={{ duration: 0.1 }}
          />

          {/* Playhead */}
          <motion.div
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-accent-primary rounded-full shadow-lg cursor-grab"
            style={{ left: `calc(${progress * 100}% - 8px)` }}
            whileHover={{ scale: 1.2 }}
            whileTap={{ scale: 0.9 }}
          />
        </div>

        {/* Time labels */}
        <div className="flex justify-between mt-2 text-xs font-mono text-gray-500">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(totalDuration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        {/* Left: Play controls */}
        <div className="flex items-center gap-2">
          {/* Reset */}
          <motion.button
            onClick={onReset}
            disabled={disabled}
            className="p-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <RotateCcw className="w-4 h-4" />
          </motion.button>

          {/* Step back */}
          <motion.button
            onClick={() => onSeek(Math.max(0, progress - 0.01))}
            disabled={disabled}
            className="p-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <ChevronLeft className="w-4 h-4" />
          </motion.button>

          {/* Play/Pause */}
          <motion.button
            onClick={isPlaying ? onPause : onPlay}
            disabled={disabled}
            className={`p-4 rounded-xl text-white transition-all ${
              isPlaying
                ? 'bg-amber-500 hover:bg-amber-600'
                : 'bg-gradient-to-r from-accent-primary to-accent-secondary hover:opacity-90'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isPlaying ? (
              <Pause className="w-5 h-5" />
            ) : (
              <Play className="w-5 h-5 ml-0.5" />
            )}
          </motion.button>

          {/* Step forward */}
          <motion.button
            onClick={() => onSeek(Math.min(1, progress + 0.01))}
            disabled={disabled}
            className="p-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <ChevronRight className="w-4 h-4" />
          </motion.button>

          {/* Single step */}
          <motion.button
            onClick={onStep}
            disabled={disabled || isPlaying}
            className="p-2.5 rounded-xl bg-gray-100 text-gray-600 hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title="Step forward one frame"
          >
            <SkipForward className="w-4 h-4" />
          </motion.button>
        </div>

        {/* Right: Speed control */}
        <div className="flex items-center gap-2">
          <Gauge className="w-4 h-4 text-gray-400" />
          <div className="flex items-center bg-gray-100 rounded-lg p-1">
            {SPEED_OPTIONS.map((speed) => (
              <motion.button
                key={speed}
                onClick={() => onSpeedChange(speed)}
                disabled={disabled}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                  playbackSpeed === speed
                    ? 'bg-white text-accent-primary shadow-sm'
                    : 'text-gray-500 hover:text-gray-700'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
                whileTap={{ scale: 0.95 }}
              >
                {speed}x
              </motion.button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
