'use client'

import { useState, useEffect } from 'react'
import { Play, Pause, RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react'

interface TrainingJob {
  job_id: string
  model_name: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  algorithm: string
  problem_type: string
  progress_percentage: number
  current_step: string
  created_at: string
  started_at?: string
  completed_at?: string
  training_metrics?: any
  validation_metrics?: any
  error_message?: string
  celery_task_id?: string
}

interface TrainingProgressProps {
  jobId?: string | null
  onJobComplete?: (jobId: string) => void
}

export default function TrainingProgress({ jobId, onJobComplete }: TrainingProgressProps) {
  const [job, setJob] = useState<TrainingJob | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(false)

  const fetchJobStatus = async (id: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/train/status/${id}`)
      
      if (response.ok) {
        const data = await response.json()
        setJob(data)
        
        // Stop polling if job is complete or failed
        if (data.status === 'completed' || data.status === 'failed') {
          setIsPolling(false)
          if (data.status === 'completed' && onJobComplete) {
            onJobComplete(id)
          }
        }
      } else {
        throw new Error('Failed to fetch job status')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status')
      setIsPolling(false)
    }
  }

  useEffect(() => {
    if (!jobId) return

    setLoading(true)
    setError(null)
    
    fetchJobStatus(jobId).finally(() => setLoading(false))
  }, [jobId])

  useEffect(() => {
    if (!jobId || !isPolling) return

    const interval = setInterval(() => {
      fetchJobStatus(jobId)
    }, 3000) // Poll every 3 seconds

    return () => clearInterval(interval)
  }, [jobId, isPolling])

  useEffect(() => {
    if (job && (job.status === 'running' || job.status === 'queued')) {
      setIsPolling(true)
    }
  }, [job])

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-600" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-600" />
      case 'running':
        return <Play className="w-5 h-5 text-blue-600" />
      case 'queued':
        return <Clock className="w-5 h-5 text-yellow-600" />
      default:
        return <RefreshCw className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-200'
      case 'running':
        return 'bg-blue-100 text-blue-800 border-blue-200'
      case 'queued':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const formatDuration = (startTime: string, endTime?: string) => {
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = Math.floor((end.getTime() - start.getTime()) / 1000)
    
    const minutes = Math.floor(duration / 60)
    const seconds = duration % 60
    
    return `${minutes}m ${seconds}s`
  }

  if (!jobId) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Clock className="w-8 h-8 mx-auto mb-2 opacity-50" />
        <p>No training job selected</p>
      </div>
    )
  }

  if (loading && !job) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-center">
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
          <p className="text-gray-600">Loading job status...</p>
        </div>
      </div>
    )
  }

  if (error && !job) {
    return (
      <div className="text-center py-8">
        <XCircle className="w-8 h-8 mx-auto mb-2 text-red-500" />
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => jobId && fetchJobStatus(jobId)}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!job) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>Job not found</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Job Header */}
      <div className={`border rounded-lg p-4 ${getStatusColor(job.status)}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon(job.status)}
            <div>
              <h3 className="font-semibold">{job.model_name}</h3>
              <p className="text-sm opacity-75">
                {job.algorithm.replace('_', ' ')} • {job.problem_type}
              </p>
            </div>
          </div>
          
          <div className="text-right text-sm">
            <div className="font-medium capitalize">{job.status}</div>
            <div className="opacity-75">Job #{job.job_id}</div>
          </div>
        </div>
      </div>

      {/* Progress Bar */}
      {(job.status === 'running' || job.status === 'queued') && (
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span>{job.current_step}</span>
            <span>{job.progress_percentage.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="bg-blue-600 h-3 rounded-full transition-all duration-300 ease-out"
              style={{ width: `${job.progress_percentage}%` }}
            />
          </div>
          {isPolling && (
            <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Updating every 3 seconds...
            </p>
          )}
        </div>
      )}

      {/* Timing Information */}
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-gray-600">Created:</span>
          <div className="font-mono">
            {new Date(job.created_at).toLocaleString()}
          </div>
        </div>
        
        {job.started_at && (
          <div>
            <span className="text-gray-600">
              {job.status === 'completed' ? 'Duration:' : 'Running for:'}
            </span>
            <div className="font-mono">
              {formatDuration(job.started_at, job.completed_at)}
            </div>
          </div>
        )}
      </div>

      {/* Training Metrics */}
      {job.training_metrics && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium mb-3">Training Metrics</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {Object.entries(job.training_metrics).map(([key, value]) => (
              <div key={key}>
                <span className="text-gray-600 capitalize">
                  {key.replace('_', ' ')}:
                </span>
                <div className="font-mono">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Validation Metrics */}
      {job.validation_metrics && (
        <div className="bg-blue-50 rounded-lg p-4">
          <h4 className="font-medium mb-3">Validation Metrics</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            {Object.entries(job.validation_metrics).map(([key, value]) => (
              <div key={key}>
                <span className="text-gray-600 capitalize">
                  {key.replace('_', ' ')}:
                </span>
                <div className="font-mono">
                  {typeof value === 'number' ? value.toFixed(4) : String(value)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {job.error_message && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h4 className="font-medium text-red-800 mb-2">Error</h4>
          <p className="text-sm text-red-700">{job.error_message}</p>
        </div>
      )}

      {/* Celery Task Info */}
      {job.celery_task_id && (
        <div className="text-xs text-gray-500">
          <span>Task ID: </span>
          <code className="bg-gray-100 px-1 rounded">{job.celery_task_id}</code>
        </div>
      )}
    </div>
  )
}
