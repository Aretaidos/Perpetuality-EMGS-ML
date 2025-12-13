// Gesture types matching the Python model output
export enum GestureType {
  INDEX_PRESS = 0,
  INDEX_RELEASE = 1,
  MIDDLE_PRESS = 2,
  MIDDLE_RELEASE = 3,
  THUMB_CLICK = 4,
  THUMB_DOWN = 5,
  THUMB_IN = 6,
  THUMB_OUT = 7,
  THUMB_UP = 8,
}

export const GESTURE_LABELS: Record<GestureType, string> = {
  [GestureType.INDEX_PRESS]: 'Index Press',
  [GestureType.INDEX_RELEASE]: 'Index Release',
  [GestureType.MIDDLE_PRESS]: 'Middle Press',
  [GestureType.MIDDLE_RELEASE]: 'Middle Release',
  [GestureType.THUMB_CLICK]: 'Thumb Click',
  [GestureType.THUMB_DOWN]: 'Thumb Down',
  [GestureType.THUMB_IN]: 'Thumb In',
  [GestureType.THUMB_OUT]: 'Thumb Out',
  [GestureType.THUMB_UP]: 'Thumb Up',
}

export const GESTURE_ICONS: Record<GestureType, string> = {
  [GestureType.INDEX_PRESS]: '👆',
  [GestureType.INDEX_RELEASE]: '👆',
  [GestureType.MIDDLE_PRESS]: '🖕',
  [GestureType.MIDDLE_RELEASE]: '🖕',
  [GestureType.THUMB_CLICK]: '👍',
  [GestureType.THUMB_DOWN]: '👇',
  [GestureType.THUMB_IN]: '👈',
  [GestureType.THUMB_OUT]: '👉',
  [GestureType.THUMB_UP]: '👆',
}

// Finger state for visualization
export interface FingerState {
  pressed: boolean
  confidence: number
  lastUpdate: number
}

// Thumb action state
export type ThumbAction = 'idle' | 'click' | 'up' | 'down' | 'in' | 'out'

export interface ThumbState {
  action: ThumbAction
  confidence: number
  lastUpdate: number
}

// Gesture event for logging
export interface GestureEvent {
  id: string
  type: GestureType
  timestamp: number
  confidence: number
}

// EMG data point
export interface EMGDataPoint {
  timestamp: number
  channels: number[]  // 7 channels
}

// Complete gesture state
export interface GestureState {
  index: FingerState
  middle: FingerState
  thumb: ThumbState
  emgBuffer: EMGDataPoint[]
  gestureLog: GestureEvent[]
  metrics: {
    latency: number
    accuracy: number
    modelConfidence: number
    samplesProcessed: number
  }
  connectionStatus: 'connected' | 'disconnected' | 'processing'
  isDemo: boolean
}

// EMG channel labels
export const EMG_CHANNEL_LABELS = [
  'Ch5 (Thumb)',
  'Ch6 (Index)',
  'Ch7 (Middle)',
  'Ch8 (Ring)',
  'Ch9 (Pinky)',
  'Ch13 (Ext1)',
  'Ch15 (Ext2)',
]

// Channel colors for visualization
export const CHANNEL_COLORS = [
  '#6366F1',  // Indigo
  '#8B5CF6',  // Purple
  '#06B6D4',  // Cyan
  '#10B981',  // Emerald
  '#F59E0B',  // Amber
  '#EC4899',  // Pink
  '#EF4444',  // Red
]
