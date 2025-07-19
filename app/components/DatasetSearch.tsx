'use client'

import { useState } from 'react'
import { Search, Database, Download, Info } from 'lucide-react'

interface Dataset {
  id: string
  name: string
  description: string
  source: 'openml' | 'kaggle' | 'local'
  size: number
  features: number
  target_column?: string
  download_url?: string
}

interface DatasetSearchProps {
  onDatasetSelect: (dataset: Dataset) => void
  selectedDataset?: Dataset | null
}

export default function DatasetSearch({ onDatasetSelect, selectedDataset }: DatasetSearchProps) {
  const [query, setQuery] = useState('')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(false)
  const [source, setSource] = useState<'all' | 'openml' | 'kaggle'>('all')

  const searchDatasets = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      // Add full URL to ensure it reaches the backend
      const response = await fetch(
        `http://localhost:8000/api/datasets/search?query=${encodeURIComponent(query)}&source=${source}&max_results=20`
      )
      
      if (response.ok) {
        const data = await response.json()
        console.log('Search results:', data)
        setDatasets(data.datasets || [])
      } else {
        console.error('Search failed with status:', response.status)
        const errorText = await response.text()
        console.error('Error details:', errorText)
        // Show mock data for demo
        setDatasets([
          {
            id: 'openml_iris',
            name: 'Iris Dataset',
            description: 'Classic classification dataset with flower measurements',
            source: 'openml',
            size: 150,
            features: 4,
            target_column: 'species'
          },
          {
            id: 'kaggle_titanic',
            name: 'Titanic Dataset',
            description: 'Passenger survival data from the Titanic disaster',
            source: 'kaggle', 
            size: 891,
            features: 11,
            target_column: 'survived'
          },
          {
            id: 'openml_wine',
            name: 'Wine Quality Dataset',
            description: 'Wine quality ratings based on chemical properties',
            source: 'openml',
            size: 1599,
            features: 11,
            target_column: 'quality'
          }
        ])
      }
    } catch (error) {
      console.error('Network error connecting to backend:', error)
      // Show mock data as fallback when backend is unreachable
      setDatasets([
        {
          id: 'mock_error',
          name: 'Backend Connection Failed',
          description: 'Unable to connect to the search service. Please check if the backend is running.',
          source: 'local',
          size: 0,
          features: 0,
          target_column: 'N/A'
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      searchDatasets()
    }
  }

  return (
    <div className="space-y-6">
      {/* Search Bar */}
      <div className="flex gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Search datasets (e.g., 'iris', 'classification', 'housing')..."
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        
        <select
          value={source}
          onChange={(e) => setSource(e.target.value as any)}
          className="px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Sources</option>
          <option value="openml">OpenML</option>
          <option value="kaggle">Kaggle</option>
        </select>
        
        <button
          onClick={searchDatasets}
          disabled={loading || !query.trim()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
          ) : (
            <Search className="w-5 h-5" />
          )}
          Search
        </button>
      </div>

      {/* Selected Dataset */}
      {selectedDataset && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-green-800 font-medium mb-2">
            <Database className="w-5 h-5" />
            Selected Dataset
          </div>
          <div className="text-sm text-green-700">
            <strong>{selectedDataset.name}</strong> ({selectedDataset.source})
            <br />
            {selectedDataset.size} rows, {selectedDataset.features} features
          </div>
        </div>
      )}

      {/* Search Results */}
      {datasets.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Search Results ({datasets.length})
          </h3>
          
          <div className="grid gap-4">
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`border rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer ${
                  selectedDataset?.id === dataset.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => onDatasetSelect(dataset)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <h4 className="font-medium text-gray-900">{dataset.name}</h4>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        dataset.source === 'openml' 
                          ? 'bg-blue-100 text-blue-800'
                          : 'bg-purple-100 text-purple-800'
                      }`}>
                        {dataset.source.toUpperCase()}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-3">{dataset.description}</p>
                    
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>{dataset.size?.toLocaleString()} rows</span>
                      <span>{dataset.features} features</span>
                      {dataset.target_column && (
                        <span>Target: {dataset.target_column}</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex gap-2 ml-4">
                    {dataset.download_url && (
                      <a
                        href={dataset.download_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-2 text-gray-400 hover:text-gray-600"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <Info className="w-4 h-4" />
                      </a>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No Results */}
      {!loading && query && datasets.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No datasets found for "{query}"</p>
          <p className="text-sm">Try different keywords or search terms</p>
        </div>
      )}
    </div>
  )
}
