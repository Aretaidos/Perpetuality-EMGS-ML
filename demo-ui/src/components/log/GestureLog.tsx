import { motion, AnimatePresence } from 'framer-motion'
import { Clock, Circle, ArrowUp, ArrowDown, ArrowLeft, ArrowRight, MousePointer, MousePointerClick } from 'lucide-react'
import { GestureEvent, GestureType, GESTURE_LABELS } from '../../types/gestures'

interface GestureLogProps {
  events: GestureEvent[]
}

// Get icon for gesture type
function getGestureIcon(type: GestureType) {
  switch (type) {
    case GestureType.INDEX_PRESS:
    case GestureType.MIDDLE_PRESS:
      return <MousePointerClick className="w-4 h-4" />
    case GestureType.INDEX_RELEASE:
    case GestureType.MIDDLE_RELEASE:
      return <MousePointer className="w-4 h-4" />
    case GestureType.THUMB_CLICK:
      return <Circle className="w-4 h-4" />
    case GestureType.THUMB_UP:
      return <ArrowUp className="w-4 h-4" />
    case GestureType.THUMB_DOWN:
      return <ArrowDown className="w-4 h-4" />
    case GestureType.THUMB_IN:
      return <ArrowLeft className="w-4 h-4" />
    case GestureType.THUMB_OUT:
      return <ArrowRight className="w-4 h-4" />
    default:
      return <Circle className="w-4 h-4" />
  }
}

// Get color for gesture type
function getGestureColor(type: GestureType): { bg: string; text: string; border: string } {
  switch (type) {
    case GestureType.INDEX_PRESS:
    case GestureType.INDEX_RELEASE:
      return { bg: 'bg-indigo-50', text: 'text-indigo-600', border: 'border-indigo-200' }
    case GestureType.MIDDLE_PRESS:
    case GestureType.MIDDLE_RELEASE:
      return { bg: 'bg-cyan-50', text: 'text-cyan-600', border: 'border-cyan-200' }
    case GestureType.THUMB_CLICK:
      return { bg: 'bg-purple-50', text: 'text-purple-600', border: 'border-purple-200' }
    case GestureType.THUMB_UP:
      return { bg: 'bg-emerald-50', text: 'text-emerald-600', border: 'border-emerald-200' }
    case GestureType.THUMB_DOWN:
      return { bg: 'bg-amber-50', text: 'text-amber-600', border: 'border-amber-200' }
    case GestureType.THUMB_IN:
      return { bg: 'bg-sky-50', text: 'text-sky-600', border: 'border-sky-200' }
    case GestureType.THUMB_OUT:
      return { bg: 'bg-pink-50', text: 'text-pink-600', border: 'border-pink-200' }
    default:
      return { bg: 'bg-gray-50', text: 'text-gray-600', border: 'border-gray-200' }
  }
}

// Format timestamp
function formatTime(timestamp: number): string {
  const date = new Date(timestamp)
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }) + '.' + String(date.getMilliseconds()).padStart(3, '0').slice(0, 2)
}

function GestureLogItem({ event }: { event: GestureEvent }) {
  const colors = getGestureColor(event.type)

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 20, scale: 0.95 }}
      transition={{ duration: 0.2 }}
      className={`flex items-center gap-3 p-3 rounded-lg border ${colors.bg} ${colors.border}`}
    >
      {/* Icon */}
      <div className={`p-2 rounded-lg bg-white/80 ${colors.text}`}>
        {getGestureIcon(event.type)}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span className={`font-medium text-sm ${colors.text}`}>
            {GESTURE_LABELS[event.type]}
          </span>
          <span className="text-xs font-mono text-gray-400">
            {event.confidence.toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-1 mt-0.5">
          <Clock className="w-3 h-3 text-gray-400" />
          <span className="text-xs text-gray-400 font-mono">
            {formatTime(event.timestamp)}
          </span>
        </div>
      </div>
    </motion.div>
  )
}

export default function GestureLog({ events }: GestureLogProps) {
  return (
    <div className="glass-panel rounded-2xl p-6 h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-800">Gesture Log</h2>
        <span className="text-sm text-gray-400">{events.length} events</span>
      </div>

      {/* Log List */}
      <div className="flex-1 overflow-y-auto space-y-2 min-h-[300px] max-h-[400px] pr-1">
        <AnimatePresence mode="popLayout">
          {events.length > 0 ? (
            events.map((event) => (
              <GestureLogItem key={event.id} event={event} />
            ))
          ) : (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex flex-col items-center justify-center h-full text-center py-12"
            >
              <div className="w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                <MousePointer className="w-8 h-8 text-gray-300" />
              </div>
              <p className="text-gray-400 text-sm">No gestures detected yet</p>
              <p className="text-gray-300 text-xs mt-1">
                Start the demo or press keys to test
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-100">
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-indigo-500" />
            <span className="text-gray-500">Index</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-cyan-500" />
            <span className="text-gray-500">Middle</span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-purple-500" />
            <span className="text-gray-500">Thumb</span>
          </div>
        </div>
      </div>
    </div>
  )
}
