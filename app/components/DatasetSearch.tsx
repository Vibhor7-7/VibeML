'use client'

import { useState, useEffect } from 'react'
import { Search, Database, Download, Info, Eye, X } from 'lucide-react'

interface Dataset {
  id: string
  name: string
  title?: string
  description: string
  source: 'openml' | 'kaggle' | 'local'
  size?: number
  features?: number
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
  const [source, setSource] = useState<'all' | 'openml' | 'kaggle' | 'local'>('all')
  const [previewDataset, setPreviewDataset] = useState<Dataset | null>(null)
  const [previewData, setPreviewData] = useState<any[]>([])
  const [previewColumns, setPreviewColumns] = useState<string[]>([])
  const [previewLoading, setPreviewLoading] = useState(false)

  // Initialize with test datasets on component mount
  useEffect(() => {
    const initializeDatasets = async () => {
      setQuery('test')
      setLoading(true)
      try {
        const response = await fetch(
          `http://localhost:8000/api/datasets/search?query=test&source=local&max_results=20`
        )
        
        if (response.ok) {
          const data = await response.json()
          console.log('Initial datasets loaded:', data)
          setDatasets(data.datasets || [])
        }
      } catch (error) {
        console.error('Failed to load initial datasets:', error)
      } finally {
        setLoading(false)
      }
    }
    
    initializeDatasets()
  }, [])

  const searchDatasets = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      // Add full URL to ensure it reaches the backend
      const response = await fetch(
        `http://localhost:8000/api/datasets/search?query=${encodeURIComponent(query || 'test')}&source=${source}&max_results=20`
      )
      
      if (response.ok) {
        const data = await response.json()
        console.log('Search results:', data)
        setDatasets(data.datasets || [])
      } else {
        console.error('Search failed with status:', response.status)
        const errorText = await response.text()
        console.error('Error details:', errorText)
        setDatasets([])
      }
    } catch (error) {
      console.error('Network error connecting to backend:', error)
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

  const generatePreviewData = (dataset: Dataset) => {
    // Generate mock preview data based on dataset type
    const commonDatasets: { [key: string]: { columns: string[], data: any[] } } = {
      'iris': {
        columns: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
        data: [
          [5.1, 3.5, 1.4, 0.2, 'setosa'],
          [4.9, 3.0, 1.4, 0.2, 'setosa'],
          [4.7, 3.2, 1.3, 0.2, 'setosa'],
          [7.0, 3.2, 4.7, 1.4, 'versicolor'],
          [6.4, 3.2, 4.5, 1.5, 'versicolor'],
          [6.3, 3.3, 6.0, 2.5, 'virginica'],
          [5.8, 2.7, 5.1, 1.9, 'virginica'],
          [7.1, 3.0, 5.9, 2.1, 'virginica']
        ]
      },
      'titanic': {
        columns: ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
        data: [
          [1, 0, 3, 'Braund, Mr. Owen Harris', 'male', 22, 1, 0, 'A/5 21171', 7.25, '', 'S'],
          [2, 1, 1, 'Cumings, Mrs. John Bradley', 'female', 38, 1, 0, 'PC 17599', 71.28, 'C85', 'C'],
          [3, 1, 3, 'Heikkinen, Miss. Laina', 'female', 26, 0, 0, 'STON/O2. 3101282', 7.92, '', 'S'],
          [4, 1, 1, 'Futrelle, Mrs. Jacques Heath', 'female', 35, 1, 0, '113803', 53.1, 'C123', 'S'],
          [5, 0, 3, 'Allen, Mr. William Henry', 'male', 35, 0, 0, '373450', 8.05, '', 'S']
        ]
      },
      'spaceship': {
        columns: ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported'],
        data: [
          ['0001_01', 'Europa', false, 'B/0/P', 'TRAPPIST-1e', 39, false, 0, 0, 0, 0, 0, 'Maham Ofracculy', false],
          ['0002_01', 'Earth', false, 'F/0/S', 'TRAPPIST-1e', 24, false, 109, 9, 25, 549, 44, 'Juanna Vines', true],
          ['0003_01', 'Europa', false, 'A/0/S', 'TRAPPIST-1e', 58, true, 43, 3576, 0, 6715, 49, 'Altark Susent', false],
          ['0004_01', 'Europa', false, 'A/0/S', 'TRAPPIST-1e', 33, false, 0, 1283, 371, 3329, 193, 'Solam Susent', false],
          ['0005_01', 'Earth', false, 'F/1/S', 'TRAPPIST-1e', 16, false, 303, 70, 151, 565, 2, 'Willy Santantines', true]
        ]
      },
      'housing': {
        columns: ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'SalePrice'],
        data: [
          [1, 60, 'RL', 65, 8450, 'Pave', null, 'Reg', 'Lvl', 'AllPub', 208500],
          [2, 20, 'RL', 80, 9600, 'Pave', null, 'Reg', 'Lvl', 'AllPub', 181500],
          [3, 60, 'RL', 68, 11250, 'Pave', null, 'IR1', 'Lvl', 'AllPub', 223500],
          [4, 70, 'RL', 60, 9550, 'Pave', null, 'IR1', 'Lvl', 'AllPub', 140000],
          [5, 60, 'RL', 84, 14260, 'Pave', null, 'IR1', 'Lvl', 'AllPub', 250000]
        ]
      },
      'census': {
        columns: ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'],
        data: [
          [39, 'State-gov', 77516, 'Bachelors', 13, 'Never-married', 'Adm-clerical', 'Not-in-family', 'White', 'Male', 2174, 0, 40, 'United-States', '<=50K'],
          [50, 'Self-emp-not-inc', 83311, 'Bachelors', 13, 'Married-civ-spouse', 'Exec-managerial', 'Husband', 'White', 'Male', 0, 0, 13, 'United-States', '<=50K'],
          [38, 'Private', 215646, 'HS-grad', 9, 'Divorced', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', 0, 0, 40, 'United-States', '<=50K'],
          [53, 'Private', 234721, '11th', 7, 'Married-civ-spouse', 'Handlers-cleaners', 'Husband', 'Black', 'Male', 0, 0, 40, 'United-States', '<=50K'],
          [28, 'Private', 338409, 'Bachelors', 13, 'Married-civ-spouse', 'Prof-specialty', 'Wife', 'Black', 'Female', 0, 0, 40, 'Cuba', '<=50K']
        ]
      },
      'digits': {
        columns: ['label', 'pixel0', 'pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7', 'pixel8', 'pixel9'],
        data: [
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
      }
    }

    // Try to match dataset name with known patterns
    const datasetName = dataset.name?.toLowerCase() || ''
    const datasetTitle = dataset.title?.toLowerCase() || dataset.description?.toLowerCase() || ''
    
    let datasetKey = Object.keys(commonDatasets).find(key => 
      datasetName.includes(key) || datasetTitle.includes(key)
    )
    
    // Handle specific dataset IDs
    if (dataset.id === 'openml_554' || datasetName.includes('iris')) {
      datasetKey = 'iris'
    } else if (datasetName.includes('spaceship') || datasetName.includes('titanic')) {
      datasetKey = datasetName.includes('spaceship') ? 'spaceship' : 'titanic'
    } else if (datasetName.includes('house') || datasetName.includes('price')) {
      datasetKey = 'housing'
    } else if (datasetName.includes('census') || datasetName.includes('adult') || dataset.id === 'openml_1590') {
      datasetKey = 'census'
    } else if (datasetName.includes('digit') || datasetName.includes('mnist')) {
      datasetKey = 'digits'
    }
    
    if (datasetKey) {
      return commonDatasets[datasetKey]
    }

    // Default generic preview data
    return {
      columns: ['feature1', 'feature2', 'feature3', 'target'],
      data: [
        [1.2, 3.4, 5.6, 'A'],
        [2.1, 4.3, 6.5, 'B'],
        [3.0, 5.2, 7.4, 'A'],
        [1.8, 2.9, 4.1, 'C'],
        [2.7, 3.8, 5.9, 'B']
      ]
    }
  }

  const handlePreview = async (dataset: Dataset, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent dataset selection
    setPreviewDataset(dataset)
    setPreviewLoading(true)
    
    try {
      // Fetch real dataset preview from backend
      const response = await fetch(
        `http://localhost:8000/api/datasets/preview/${dataset.source}/${encodeURIComponent(dataset.name)}?max_rows=10`
      )
      
      if (response.ok) {
        const previewInfo = await response.json()
        console.log('Dataset preview:', previewInfo)
        
        setPreviewColumns(previewInfo.columns)
        setPreviewData(previewInfo.sample_data)
      } else {
        console.error('Failed to fetch dataset preview:', await response.text())
        setPreviewColumns(['Error'])
        setPreviewData([{ Error: 'Failed to load preview. The backend may be unavailable.' }])
      }
    } catch (error) {
      console.error('Error fetching dataset preview:', error)
      setPreviewColumns(['Error'])
      setPreviewData([{ Error: 'Failed to load preview. The backend may be unavailable.' }])
    } finally {
      setPreviewLoading(false)
    }
  }

  const closePreview = () => {
    setPreviewDataset(null)
    setPreviewData([])
    setPreviewColumns([])
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
          <option value="local">Local Test Datasets</option>
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
                          : dataset.source === 'local'
                          ? 'bg-green-100 text-green-800'
                          : 'bg-purple-100 text-purple-800'
                      }`}>
                        {dataset.source.toUpperCase()}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-600 mb-3">{dataset.description}</p>
                    
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      {dataset.size && <span>{dataset.size.toLocaleString()} rows</span>}
                      {dataset.features && <span>{dataset.features} features</span>}
                      {dataset.target_column && (
                        <span>Target: {dataset.target_column}</span>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex gap-2 ml-4">
                    <button
                      onClick={(e) => handlePreview(dataset, e)}
                      className="p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-md transition-colors"
                      title="Preview Dataset"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
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

      {/* Preview Modal */}
      {previewDataset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-blue-600" />
                  Dataset Preview: {previewDataset.name}
                </h3>
                <p className="text-sm text-gray-600 mt-1">{previewDataset.description}</p>
              </div>
              <button
                onClick={closePreview}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              {previewLoading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="w-8 h-8 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                  <span className="ml-3 text-gray-600">Loading preview...</span>
                </div>
              ) : (
                <>
                  <div className="mb-4">
                    <p className="text-sm text-gray-600">
                      Preview showing sample data from the {previewDataset.name} dataset
                    </p>
                  </div>

                  {/* Preview Table */}
                  <div className="overflow-hidden rounded-lg border border-gray-300 bg-white">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="bg-gray-50 border-b border-gray-200">
                            {previewColumns.map((column, index) => (
                              <th key={index} className="px-4 py-3 text-left text-gray-900 text-sm font-medium">
                                {column}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {previewData.map((row, index) => (
                            <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                              {previewColumns.map((column, colIndex) => (
                                <td key={colIndex} className="px-4 py-3 text-gray-700 text-sm">
                                  {row[column] !== null && row[column] !== undefined ? String(row[column]) : '-'}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-between">
                    <p className="text-sm text-gray-600">
                      Showing {previewData.length} sample rows • {previewDataset.size?.toLocaleString()} total rows • {previewDataset.features} features
                    </p>
                    <button
                      onClick={() => {
                        onDatasetSelect(previewDataset)
                        closePreview()
                      }}
                      className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors"
                    >
                      Select This Dataset
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
