'use client'

import { useState } from 'react'
import { Upload, Play, Download, Settings, BarChart3, Database, Code, Globe, Target, FileText, Search } from 'lucide-react'
import Image from 'next/image'
import DatasetSearch from './components/DatasetSearch'
import TrainingProgress from './components/TrainingProgress'
import ModelList from './components/ModelList'
import PredictionForm from './components/PredictionForm'

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('search')
  const [isTraining, setIsTraining] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [previewData, setPreviewData] = useState<any[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [autoMode, setAutoMode] = useState(false)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [selectedDataset, setSelectedDataset] = useState<any>(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [problemType, setProblemType] = useState('classification')
  const [datasetColumns, setDatasetColumns] = useState<string[]>([])
  const [modelConfig, setModelConfig] = useState({
    n_estimators: '',
    max_depth: '',
    criterion: 'gini'
  })

  const models = [
    {
      id: 'logistic_regression',
      name: 'Logistic Regression',
      description: 'A linear model for binary classification.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCJbbw6xr3tTO7843lVsaL-ryYCcUzWdMjLBowHMwzfe6vCk-FqaZPyPxQ2BOxCEUckzZQ1SkSooFT1T13D9rSA_Ie9p950kT1zcY3MiSrtm2tBZv27te29qKPUq8KVKwLRJdoIDkuUzkI1sWKbRLzEP2f7jOdhSFHZ6-aPfp42_MC_qSC7PXbNf8kd8SBnkz7U6YMPkX1LS0iOguFKDS3Ti55BwdRbWRMedgZflDgzxPNtCSKwDlwNDRE83cWogxIoNbAMWDPqp3c'
    },
    {
      id: 'random_forest',
      name: 'Random Forest',
      description: 'An ensemble of decision trees for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDSwdzR4PH6hmYErHh5YAQis-yPs3rYLz_ChqOsaDvEqvXUhrXNOuru6Dj-EmTfR_-e8pYK62G0vBadjLM6ASVfZ8qDz8t38iNnAimyYDPl75lUdRBsXFpQ03_iPknmI34oCDGEgPCSSkMi2_CZPVCnDoDTvE4tHD8x9pB5l8kGvBIEYz1oNhM33zuUf6vuu1Gk3O9YyGPrPrYHzAJSlfKNHAQrpzwEUDcPy8rzj8N--atSEtMvvBLcIID-5fVUqlN9zxE5dAn-iGk'
    },
    {
      id: 'svm',
      name: 'Support Vector Machine',
      description: 'A powerful model for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuA5OkzBuRr5sABjvDhU7amWguOC-tl0RrjGPXOY2wzTajRSGQJny8iyuEqLj4tQ19meuTCCM-7dQjoQKJZ5Ug5s0blVebv5CvRhM_SsGwSCzHxY2b8c2MNBE1-ejSezKpn_rSWoy_qYvqZUykCzP21tDJgNO8ybf3qrBYk7TOXjAZqxUwWbSetX4mUuiF7Tx-7rlDLIBH40-qUuECtRD5dj6KhOXrzCfubCIGPOANU-qQTMny9HaUDJAEo3P5K46hloOgybIySZdBA'
    },
    {
      id: 'knn',
      name: 'K-Nearest Neighbors',
      description: 'A simple, non-parametric model for classification and regression.',
      image: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBMdKc85t7mUXtC9QaQHRxcuYl3qC9l5MAapFmdlKXS8xQETSIvNWrCY7QL8Lu1VmqoP3QWPhCIRQVqklbaq5TkkGtUH43_lv8-eoJqMkJZUBbTtyiDrGFTv0RSaDkOJ7EtZN0rVUbE7wy-WeXo1xDP5qbjYbl-Z7kiXrEwuG3vk14JufdBXfU2Z9XtbYvXPEvL-onBa7URUv2UXskE0wO0S_ZaxRAY1u_Y_dC2FmMnS5sc7AmSeFjjQm2a25m7I1Fmsa1eSpCFRyU'
    }
  ]

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    console.log('File uploaded:', file?.name, 'Type:', file?.type)
    if (file && file.type === 'text/csv') {
      setUploadedFile(file)
      setSelectedDataset(null) // Clear selected dataset when uploading a file
      setTargetColumn('') // Reset target column when switching datasets
      
      // Parse CSV to extract column names and data
      const reader = new FileReader()
      reader.onload = (e) => {
        const text = e.target?.result as string
        console.log('File content preview:', text.substring(0, 200))
        const lines = text.split('\n').filter(line => line.trim()) // Remove empty lines
        
        if (lines.length > 0) {
          const headers = lines[0].split(',').map(col => col.trim().replace(/"/g, ''))
          console.log('CSV columns extracted:', headers)
          setDatasetColumns(headers)
          setTargetColumn('') // Reset target column when new file is uploaded
          
          // Parse the actual data rows
          const dataRows = lines.slice(1, Math.min(11, lines.length)) // Take up to 10 data rows
          const parsedData = dataRows.map((line, index) => {
            const values = line.split(',').map(val => val.trim().replace(/"/g, ''))
            const rowData: any = { _rowIndex: index }
            headers.forEach((header, colIndex) => {
              rowData[header] = values[colIndex] || ''
            })
            return rowData
          })
          
          console.log('Parsed CSV data:', parsedData)
          setPreviewData(parsedData)
        }
      }
      reader.readAsText(file)
      
      setShowPreview(true)
    }
  }

  const handlePreviewConfirm = () => {
    setShowPreview(false)
    setActiveTab('train')
    console.log('Data confirmed, proceeding to training...')
  }

  const handleReupload = () => {
    setUploadedFile(null)
    setSelectedDataset(null) // Clear selected dataset when reuploading
    setShowPreview(false)
    setPreviewData([])
    setDatasetColumns([]) // Clear columns
    setTargetColumn('') // Clear target column
  }

  const fetchDatasetColumns = async (dataset: any) => {
    try {
      console.log('Fetching columns for dataset:', dataset)
      
      // For external datasets, ensure proper encoding of dataset name
      const encodedDatasetName = encodeURIComponent(dataset.name)
      
      // Fetch dataset info from backend
      const response = await fetch(
        `http://localhost:8000/api/datasets/preview/${dataset.source}/${encodedDatasetName}?max_rows=5`
      )
      
      if (response.ok) {
        const previewInfo = await response.json()
        console.log('Dataset preview info:', previewInfo)
        
        // Also update preview data if available
        if (previewInfo.sample_data && previewInfo.sample_data.length > 0) {
          setPreviewData(previewInfo.sample_data)
        }
        
        // Set potential target columns
        if (previewInfo.potential_target_columns && previewInfo.potential_target_columns.length > 0) {
          setTargetColumn(previewInfo.potential_target_columns[0])
        }
        
        return previewInfo.columns
      } else {
        const errorText = await response.text()
        console.error('Failed to fetch dataset columns:', errorText)
        
        // For external datasets that fail to preview, try to use known good columns
        if (dataset.source === 'openml' && dataset.name === '61') {
          // Known OpenML Iris dataset columns
          return ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
        }
        
        // Default fallback
        return ['feature1', 'feature2', 'feature3', 'target']
      }
    } catch (error) {
      console.error('Error fetching dataset columns:', error)
      
      // For external datasets that fail to preview, try to use known good columns
      if (dataset.source === 'openml' && dataset.name === '61') {
        // Known OpenML Iris dataset columns
        return ['sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'class']
      }
      
      return ['feature1', 'feature2', 'feature3', 'target']
    }
  }

  const handleDatasetSelect = async (dataset: any) => {
    console.log('Dataset selected:', dataset)
    setSelectedDataset(dataset)
    setUploadedFile(null) // Clear uploaded file when selecting a dataset
    setTargetColumn('') // Reset target column when switching datasets
    
    // Fetch actual column information
    const columns = await fetchDatasetColumns(dataset)
    console.log('Fetched columns:', columns)
    setDatasetColumns(columns)
  }

  const handleTestExternalDataset = async () => {
    // Add a test OpenML Iris dataset that we know works
    const testDataset = {
      id: 61,
      name: '61',
      source: 'openml',
      title: 'OpenML Iris Dataset (Test)',
      description: 'Classic iris dataset for testing external dataset functionality'
    }
    
    console.log('Loading test external dataset...')
    await handleDatasetSelect(testDataset)
  }

  const handleTrain = async () => {
    try {
      setIsTraining(true)
      setLogs(['Preparing training configuration...'])
      
      // Validate required fields
      if (!targetColumn) {
        throw new Error('Please select a target column')
      }
      
      // Handle different dataset sources
      let datasetId = 'unknown'
      let datasetSource = 'upload'
      
      if (selectedDataset) {
        // For selected datasets from search, use the dataset info
        datasetSource = selectedDataset.source
        datasetId = selectedDataset.name || selectedDataset.id || 'unknown'
        
        // For OpenML datasets, ensure we have just the numeric ID
        if (datasetSource === 'openml' && datasetId.startsWith('openml_')) {
          datasetId = datasetId.replace('openml_', '')
        }
        
        setLogs(prev => [...prev, `Using ${datasetSource} dataset: ${datasetId}`])
        
        // Pre-download the dataset to verify it exists and get info
        try {
          const downloadResponse = await fetch(
            `http://localhost:8000/api/datasets/download/${datasetSource}/${encodeURIComponent(datasetId)}`,
            { method: 'GET' }
          )
          
          if (!downloadResponse.ok) {
            throw new Error('Failed to prepare dataset for training')
          }
          
          const downloadResult = await downloadResponse.json()
          console.log('Dataset prepared:', downloadResult)
          setLogs(prev => [...prev, `Dataset prepared: ${downloadResult.shape[0]} rows, ${downloadResult.shape[1]} columns`])
          
        } catch (error) {
          console.error('Dataset preparation error:', error)
          throw new Error(`Failed to prepare dataset: ${error.message}`)
        }
        
      } else if (uploadedFile) {
        // Handle file upload
        setLogs(prev => [...prev, 'Uploading dataset...'])
        
        const formData = new FormData()
        formData.append('file', uploadedFile)
        formData.append('name', uploadedFile.name.split('.')[0])
        formData.append('target_column', targetColumn)
        
        const uploadResponse = await fetch('http://localhost:8000/api/import/upload', {
          method: 'POST',
          body: formData
        })
        
        if (!uploadResponse.ok) {
          const errorData = await uploadResponse.json()
          throw new Error(errorData.detail || 'Failed to upload dataset')
        }
        
        const uploadResult = await uploadResponse.json()
        datasetId = uploadResult.dataset_id
        datasetSource = 'upload'
        setLogs(prev => [...prev, `Dataset uploaded with ID: ${datasetId}`])
      } else {
        throw new Error('No dataset selected or uploaded')
      }
      
      // Prepare training configuration
      const trainingConfig = {
        model_name: `model_${Date.now()}`,
        dataset_id: datasetId,
        dataset_source: datasetSource,
        target_column: targetColumn,
        problem_type: problemType,
        algorithm: autoMode ? 'auto' : selectedModel,
        test_size: 0.2,
        auto_hyperparameter_tuning: autoMode,
        hyperparameters: selectedModel === 'random_forest' ? {
          n_estimators: modelConfig.n_estimators ? parseInt(modelConfig.n_estimators) : 100,
          max_depth: modelConfig.max_depth ? parseInt(modelConfig.max_depth) : null
        } : {}
      }
      
      setLogs(prev => [...prev, 'Starting training job...'])
      console.log('Training config:', trainingConfig)
      
      // Start training job
      const response = await fetch('http://localhost:8000/api/train/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingConfig)
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Training failed to start')
      }
      
      const result = await response.json()
      setCurrentJobId(result.job.job_id)
      setLogs(prev => [...prev, `Training job started with ID: ${result.job.job_id}`])
      
    } catch (error) {
      console.error('Training error:', error)
      setLogs(prev => [...prev, `Error: ${error.message}`])
      setIsTraining(false)
    }
  }

  const navItems = [
    { id: 'search', label: 'Search Datasets', icon: Search, color: 'purple' },
    { id: 'upload', label: 'Upload Data', icon: Upload, color: 'blue' },
    { id: 'train', label: 'Train Model', icon: Play, color: 'green' },
    { id: 'evaluate', label: 'Evaluate', icon: Target, color: 'purple' },
    { id: 'predict', label: 'Predict', icon: BarChart3, color: 'orange' },
    { id: 'export', label: 'Export', icon: FileText, color: 'cyan' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-green-900 flex">
      {/* Left Sidebar Navigation */}
      <div className="w-64 bg-gray-800/80 backdrop-blur-sm border-r border-gray-700 flex flex-col">
        {/* Logo Section */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 relative">
              <Image
                src="/Screenshot 2025-07-19 at 10.33.37 AM.png"
                alt="VibeML Logo"
                width={40}
                height={40}
                className="rounded-lg object-contain"
              />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">VibeML</h1>
              <span className="text-xs text-gray-300">No-Code AutoML</span>
            </div>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 p-4">
          <div className="space-y-2">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = activeTab === item.id
              
              // Check if tab should be disabled
              const hasDataset = selectedDataset || uploadedFile
              const isDataTab = item.id === 'search' || item.id === 'upload'
              const isDisabled = !isDataTab && !hasDataset
              
              const colorClasses = {
                blue: isActive ? 'bg-blue-600/20 text-blue-300 border-blue-500/30' : isDisabled ? 'text-gray-500 cursor-not-allowed' : 'text-gray-300 hover:text-blue-300 hover:bg-blue-600/10',
                green: isActive ? 'bg-green-600/20 text-green-300 border-green-500/30' : isDisabled ? 'text-gray-500 cursor-not-allowed' : 'text-gray-300 hover:text-green-300 hover:bg-green-600/10',
                purple: isActive ? 'bg-purple-600/20 text-purple-300 border-purple-500/30' : isDisabled ? 'text-gray-500 cursor-not-allowed' : 'text-gray-300 hover:text-purple-300 hover:bg-purple-600/10',
                orange: isActive ? 'bg-orange-600/20 text-orange-300 border-orange-500/30' : isDisabled ? 'text-gray-500 cursor-not-allowed' : 'text-gray-300 hover:text-orange-300 hover:bg-orange-600/10',
                cyan: isActive ? 'bg-cyan-600/20 text-cyan-300 border-cyan-500/30' : isDisabled ? 'text-gray-500 cursor-not-allowed' : 'text-gray-300 hover:text-cyan-300 hover:bg-cyan-600/10',
              }
              
              return (
                <button
                  key={item.id}
                  onClick={() => {
                    if (!isDisabled) {
                      setActiveTab(item.id)
                    }
                  }}
                  disabled={isDisabled}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${
                    isActive ? `border ${colorClasses[item.color]}` : colorClasses[item.color]
                  } ${isDisabled ? 'opacity-50' : ''}`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                  {isDisabled && (
                    <span className="ml-auto text-xs bg-gray-600 text-gray-300 px-2 py-1 rounded">
                      Locked
                    </span>
                  )}
                </button>
              )
            })}
          </div>
        </nav>

        {/* Quick Stats in Sidebar */}
        <div className="p-4 border-t border-gray-700">
          <h3 className="text-sm font-semibold mb-3 text-gray-300">Quick Stats</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Models</span>
              <span className="text-blue-400 font-medium">0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Datasets</span>
              <span className={`font-medium ${(selectedDataset || uploadedFile) ? 'text-green-400' : 'text-gray-500'}`}>
                {(selectedDataset || uploadedFile) ? '1' : '0'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">APIs</span>
              <span className="text-cyan-400 font-medium">0</span>
            </div>
          </div>
          
          {/* Dataset Status */}
          {(selectedDataset || uploadedFile) && (
            <div className="mt-4 p-3 bg-green-900/20 border border-green-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-green-300 text-xs font-medium">Dataset Ready</span>
              </div>
              <p className="text-green-200 text-xs">
                {selectedDataset ? selectedDataset.name : uploadedFile?.name}
              </p>
            </div>
          )}
          
          {/* Instructions when no dataset */}
          {!selectedDataset && !uploadedFile && (
            <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-500/30 rounded-lg">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                <span className="text-yellow-300 text-xs font-medium">Get Started</span>
              </div>
              <p className="text-yellow-200 text-xs">
                Search or upload a dataset to unlock all features
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Main Panel */}
        <main className="flex-1 p-8 overflow-y-auto">
          <div className="max-w-6xl mx-auto">
            {activeTab === 'search' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-2xl font-semibold flex items-center text-white">
                    <Search className="w-6 h-6 mr-3 text-purple-400" />
                    Search Datasets
                  </h2>
                  <button
                    onClick={handleTestExternalDataset}
                    className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-all font-medium text-sm"
                  >
                    Test OpenML Iris
                  </button>
                </div>
                <DatasetSearch
                  onDatasetSelect={(dataset) => {
                    handleDatasetSelect(dataset)
                    setActiveTab('train')
                  }}
                />
              </div>
            )}

            {activeTab === 'upload' && (
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                  <Database className="w-6 h-6 mr-3 text-blue-400" />
                  {showPreview ? 'Data Preview & Validation' : 'Upload Your Dataset'}
                </h2>
                
                {!showPreview ? (
                  <>
                    <div className="border-2 border-dashed border-gray-600 rounded-lg p-12 text-center hover:border-blue-400 transition-colors cursor-pointer bg-gray-900/30">
                      <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="csv-upload"
                      />
                      <label htmlFor="csv-upload" className="cursor-pointer">
                        <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                        <h3 className="text-xl font-medium text-white mb-2">
                          Drop your CSV file here
                        </h3>
                        <p className="text-gray-400 mb-6">
                          or click to browse files
                        </p>
                        <span className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-md hover:from-blue-700 hover:to-blue-800 transition-all font-medium inline-block">
                          Choose File
                        </span>
                      </label>
                    </div>
                    <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Supported Formats</h4>
                        <p className="text-sm text-gray-300">CSV files up to 100MB</p>
                      </div>
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Auto-Detection</h4>
                        <p className="text-sm text-gray-300">Automatically detects data types</p>
                      </div>
                      <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                        <h4 className="font-medium text-white text-lg mb-2">Preprocessing</h4>
                        <p className="text-sm text-gray-300">Handles missing values</p>
                      </div>
                    </div>
                  </>
                ) : (
                  <>
                    <p className="text-gray-300 text-sm mb-6">
                      Here's a sample of your dataset. Please verify the structure and content before proceeding.
                    </p>
                    
                    {/* Data Preview Table */}
                    <div className="overflow-hidden rounded-lg border border-gray-600 bg-gray-800/30 mb-4">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="bg-gray-700/50">
                              {datasetColumns.map((column, index) => (
                                <th key={index} className="px-4 py-3 text-left text-white text-sm font-medium">
                                  {column}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {previewData.map((row, index) => (
                              <tr key={index} className="border-t border-gray-600">
                                {datasetColumns.map((column, colIndex) => (
                                  <td key={colIndex} className="px-4 py-3 text-gray-300 text-sm">
                                    {row[column] || ''}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    
                    <p className="text-blue-400 text-sm mb-6 px-4">
                      {previewData.length} of {previewData.length} rows shown
                    </p>
                    
                    {/* Action Buttons */}
                    <div className="flex justify-end gap-4">
                      <button
                        onClick={handlePreviewConfirm}
                        className="bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-3 rounded-lg hover:from-blue-700 hover:to-blue-800 transition-all font-medium"
                      >
                        Looks Good
                      </button>
                      <button
                        onClick={handleReupload}
                        className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-all font-medium"
                      >
                        Re-upload CSV
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'train' && (
              <>
                {!selectedDataset && !uploadedFile ? (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-12 border border-gray-700 text-center">
                    <div className="max-w-md mx-auto">
                      <Play className="w-16 h-16 text-gray-500 mx-auto mb-6" />
                      <h2 className="text-2xl font-semibold text-white mb-4">Dataset Required</h2>
                      <p className="text-gray-300 mb-8">
                        To start training a model, you need to first search for a dataset or upload your own data file.
                      </p>
                      <div className="flex gap-4 justify-center">
                        <button
                          onClick={() => setActiveTab('search')}
                          className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-all font-medium"
                        >
                          Search Datasets
                        </button>
                        <button
                          onClick={() => setActiveTab('upload')}
                          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-all font-medium"
                        >
                          Upload Data
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                {/* Header */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                    <Play className="w-6 h-6 mr-3 text-green-400" />
                    Train Model
                  </h2>
                  
                  {selectedDataset && (
                    <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4 mb-6">
                      <h3 className="text-blue-300 font-medium mb-2">Selected Dataset</h3>
                      <p className="text-white">{selectedDataset.name}</p>
                      <p className="text-gray-300 text-sm">{selectedDataset.description}</p>
                    </div>
                  )}
                  
                  {uploadedFile && (
                    <div className="bg-green-900/20 border border-green-500/30 rounded-lg p-4 mb-6">
                      <h3 className="text-green-300 font-medium mb-2">Uploaded File</h3>
                      <p className="text-white">{uploadedFile.name}</p>
                      <p className="text-gray-300 text-sm">Ready for training</p>
                    </div>
                  )}
                  
                  {!selectedDataset && !uploadedFile && (
                    <div className="bg-gray-700/30 rounded-lg p-4 text-center">
                      <p className="text-gray-300 mb-4">No dataset selected. Please search for a dataset or upload your own file.</p>
                      <div className="flex gap-4 justify-center">
                        <button
                          onClick={() => setActiveTab('search')}
                          className="bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 transition-all"
                        >
                          Search Datasets
                        </button>
                        <button
                          onClick={() => setActiveTab('upload')}
                          className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-all"
                        >
                          Upload File
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                {/* Training Form and Progress */}
                {(selectedDataset || uploadedFile) && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Training Configuration */}
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                      <h3 className="text-xl font-semibold text-white mb-6">Training Configuration</h3>
                      
                      {/* Auto Mode Toggle */}
                      <div className="bg-gray-700/30 rounded-lg p-4 mb-6 border border-gray-600">
                        <div className="flex items-center justify-between">
                          <div>
                            <h4 className="text-white font-medium mb-1">Auto Mode</h4>
                            <p className="text-gray-300 text-sm">
                              Automatically select the best model and parameters
                            </p>
                          </div>
                          <label className="relative flex h-8 w-14 cursor-pointer items-center rounded-full bg-gray-600 p-1 transition-colors has-[:checked]:bg-green-600">
                            <div className={`h-6 w-6 rounded-full bg-white transition-transform ${autoMode ? 'translate-x-6' : ''}`}></div>
                            <input
                              type="checkbox"
                              className="sr-only"
                              checked={autoMode}
                              onChange={(e) => setAutoMode(e.target.checked)}
                            />
                          </label>
                        </div>
                      </div>

                      {/* Dataset Configuration */}
                      <div className="space-y-4 mb-6">
                        {/* Debug info */}
                        <div className="text-xs text-gray-500 bg-gray-800 p-2 rounded">
                          Debug - Dataset Columns: {JSON.stringify(datasetColumns)}
                        </div>
                        
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Target Column *
                          </label>
                          <select
                            value={targetColumn}
                            onChange={(e) => setTargetColumn(e.target.value)}
                            className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white"
                          >
                            <option value="">Select target column...</option>
                            {datasetColumns.map((column) => {
                              console.log('Rendering column option:', column)
                              return (
                                <option key={column} value={column}>
                                  {column}
                                </option>
                              )
                            })}
                          </select>
                          <p className="text-gray-400 text-xs mt-1">
                            The column you want to predict
                          </p>
                        </div>

                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Problem Type
                          </label>
                          <select
                            value={problemType}
                            onChange={(e) => setProblemType(e.target.value)}
                            className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white"
                          >
                            <option value="classification">Classification</option>
                            <option value="regression">Regression</option>
                          </select>
                          <p className="text-gray-400 text-xs mt-1">
                            Type of machine learning problem
                          </p>
                        </div>
                      </div>

                      {/* Model Selection */}
                      {!autoMode && (
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                              Model Type
                            </label>
                            <select
                              value={selectedModel}
                              onChange={(e) => setSelectedModel(e.target.value)}
                              className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white"
                            >
                              <option value="">Select model...</option>
                              {models.map((model) => (
                                <option key={model.id} value={model.id}>
                                  {model.name}
                                </option>
                              ))}
                            </select>
                          </div>

                          {selectedModel === 'random_forest' && (
                            <>
                              <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">
                                  n_estimators
                                </label>
                                <input
                                  type="number"
                                  value={modelConfig.n_estimators}
                                  onChange={(e) => setModelConfig({...modelConfig, n_estimators: e.target.value})}
                                  placeholder="100"
                                  className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white placeholder-gray-400"
                                />
                              </div>
                              <div>
                                <label className="block text-sm font-medium text-gray-300 mb-2">
                                  max_depth
                                </label>
                                <input
                                  type="number"
                                  value={modelConfig.max_depth}
                                  onChange={(e) => setModelConfig({...modelConfig, max_depth: e.target.value})}
                                  placeholder="None"
                                  className="w-full p-3 border border-gray-600 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent bg-gray-700 text-white placeholder-gray-400"
                                />
                              </div>
                            </>
                          )}
                        </div>
                      )}

                      {/* Start Training Button */}
                      <div className="mt-6">
                        <button
                          onClick={handleTrain}
                          disabled={(!selectedModel && !autoMode) || !targetColumn}
                          className={`w-full px-6 py-3 rounded-lg font-medium transition-all ${
                            ((!selectedModel && !autoMode) || !targetColumn) || isTraining
                              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                              : 'bg-gradient-to-r from-green-600 to-green-700 text-white hover:from-green-700 hover:to-green-800'
                          }`}
                        >
                          {isTraining ? 'Training in Progress...' : 'Start Training'}
                        </button>
                      </div>
                    </div>

                    {/* Training Progress */}
                    <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                      <h3 className="text-xl font-semibold text-white mb-6">Training Progress</h3>
                      <TrainingProgress 
                        jobId={currentJobId}
                        onJobComplete={(jobId) => {
                          setIsTraining(false)
                          setCurrentJobId(null)
                          setActiveTab('evaluate')
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
                )}
              </>
            )}

            {activeTab === 'evaluate' && (
              <>
                {!selectedDataset && !uploadedFile ? (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-12 border border-gray-700 text-center">
                    <div className="max-w-md mx-auto">
                      <Target className="w-16 h-16 text-gray-500 mx-auto mb-6" />
                      <h2 className="text-2xl font-semibold text-white mb-4">Dataset Required</h2>
                      <p className="text-gray-300 mb-8">
                        To evaluate models, you need to first have a dataset and train a model.
                      </p>
                      <div className="flex gap-4 justify-center">
                        <button
                          onClick={() => setActiveTab('search')}
                          className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-all font-medium"
                        >
                          Search Datasets
                        </button>
                        <button
                          onClick={() => setActiveTab('upload')}
                          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-all font-medium"
                        >
                          Upload Data
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                {/* Header Section */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <div className="flex justify-between items-start mb-6">
                    <div>
                      <h2 className="text-2xl font-semibold flex items-center text-white mb-2">
                        <Target className="w-6 h-6 mr-3 text-purple-400" />
                        Model Analytics Dashboard
                      </h2>
                      <p className="text-gray-300 text-sm">
                        Explore detailed insights into your trained model's performance and dataset characteristics.
                      </p>
                    </div>
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Performance Metrics</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Accuracy</h4>
                      <div className="text-2xl font-bold text-white mb-1">92%</div>
                      <p className="text-green-400 text-base font-medium">+2%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Precision</h4>
                      <div className="text-2xl font-bold text-white mb-1">88%</div>
                      <p className="text-red-400 text-base font-medium">-1%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">Recall</h4>
                      <div className="text-2xl font-bold text-white mb-1">90%</div>
                      <p className="text-green-400 text-base font-medium">+3%</p>
                    </div>
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">F1 Score</h4>
                      <div className="text-2xl font-bold text-white mb-1">89%</div>
                      <p className="text-green-400 text-base font-medium">+1%</p>
                    </div>
                  </div>
                  
                  {/* ROC-AUC Score */}
                  <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
                    <div className="bg-gray-700/50 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-base font-medium text-white mb-2">ROC-AUC Score</h4>
                      <div className="text-2xl font-bold text-white">0.95</div>
                    </div>
                  </div>
                </div>

                {/* Model Overview */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Model Overview</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Model Type</p>
                      <p className="text-white text-sm">Classification</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Training Time</p>
                      <p className="text-white text-sm">2 hours 30 minutes</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Number of Iterations/Epochs</p>
                      <p className="text-white text-sm">100 epochs</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Algorithm Used</p>
                      <p className="text-white text-sm">Gradient Boosting</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Dataset Name</p>
                      <p className="text-white text-sm">Customer Retention Dataset</p>
                    </div>
                    <div className="flex flex-col gap-1 border-t border-gray-600 py-4">
                      <p className="text-gray-400 text-sm">Dataset Size</p>
                      <p className="text-white text-sm">10,000 rows</p>
                    </div>
                  </div>
                </div>

                {/* Interactive Charts */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl border border-gray-700">
                  <div className="p-6 pb-0">
                    <h3 className="text-xl font-semibold text-white mb-6">Interactive Charts</h3>
                  </div>
                  
                  {/* Chart Tabs */}
                  <div className="border-b border-gray-600">
                    <div className="flex px-6 gap-8">
                      <button className="flex flex-col items-center justify-center border-b-2 border-blue-400 text-white pb-3 pt-4">
                        <span className="text-white text-sm font-bold">ROC Curve</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Precision-Recall</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Feature Importance</span>
                      </button>
                      <button className="flex flex-col items-center justify-center border-b-2 border-transparent text-gray-400 pb-3 pt-4 hover:text-white">
                        <span className="text-gray-400 text-sm font-bold">Loss vs Epoch</span>
                      </button>
                    </div>
                  </div>
                  
                  {/* Chart Content */}
                  <div className="p-6">
                    <div className="bg-gray-700/30 rounded-lg p-6 border border-gray-600">
                      <div className="flex justify-between items-start mb-4">
                        <div>
                          <h4 className="text-white font-medium mb-1">ROC Curve</h4>
                          <div className="text-2xl font-bold text-white mb-2">0.95</div>
                          <div className="flex gap-2 items-center">
                            <span className="text-gray-300 text-sm">Overall</span>
                            <span className="text-green-400 text-sm font-medium">+2%</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Placeholder Chart */}
                      <div className="h-48 bg-gray-800/50 rounded-lg border border-gray-600 flex items-center justify-center relative overflow-hidden">
                        {/* Simple SVG curve simulation */}
                        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 200">
                          <defs>
                            <linearGradient id="rocGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="rgb(59, 130, 246)" stopOpacity="0.3"/>
                              <stop offset="100%" stopColor="rgb(59, 130, 246)" stopOpacity="0.1"/>
                            </linearGradient>
                          </defs>
                          <path
                            d="M 20 180 Q 100 160 200 100 T 380 20"
                            stroke="rgb(59, 130, 246)"
                            strokeWidth="2"
                            fill="none"
                          />
                          <path
                            d="M 20 180 Q 100 160 200 100 T 380 20 L 380 180 L 20 180 Z"
                            fill="url(#rocGradient)"
                          />
                        </svg>
                        <div className="absolute bottom-2 left-0 right-0 flex justify-between px-4 text-xs text-gray-400">
                          <span>0.0</span>
                          <span>0.2</span>
                          <span>0.4</span>
                          <span>0.6</span>
                          <span>0.8</span>
                          <span>1.0</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Dataset Summary */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Dataset Summary</h3>
                  <p className="text-white text-base mb-6">Key insights about the dataset used for training:</p>
                  
                  <div className="space-y-4">
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Null Values</h4>
                      <p className="text-gray-300 text-sm">
                        15% of the dataset contains missing values, primarily in the 'Age' and 'Income' columns.
                      </p>
                    </div>
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Dominant Class</h4>
                      <p className="text-gray-300 text-sm">
                        The 'Retention' class is the dominant class, representing 70% of the dataset.
                      </p>
                    </div>
                    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-1">Class Imbalance</h4>
                      <p className="text-gray-300 text-sm">
                        The dataset exhibits a moderate class imbalance, with a ratio of approximately 2:1 between the 'Retention' and 'No Retention' classes.
                      </p>
                    </div>
                  </div>

                  {/* Distribution Charts */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <div className="bg-gray-700/30 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-2">Distribution of Categorical Variable A</h4>
                      <div className="text-2xl font-bold text-white mb-2">60%</div>
                      <div className="flex gap-2 items-center mb-4">
                        <span className="text-gray-300 text-sm">Overall</span>
                        <span className="text-green-400 text-sm font-medium">+5%</span>
                      </div>
                      
                      {/* Bar Chart */}
                      <div className="h-32 flex items-end justify-center gap-4">
                        <div className="flex flex-col items-center">
                          <div className="w-12 bg-blue-500 border-t-2 border-blue-400" style={{height: '60%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">Category 1</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-12 bg-blue-500 border-t-2 border-blue-400" style={{height: '40%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">Category 2</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-700/30 p-6 rounded-lg border border-gray-600">
                      <h4 className="text-white font-medium mb-2">Distribution of Categorical Variable B</h4>
                      <div className="text-2xl font-bold text-white mb-2">40%</div>
                      <div className="flex gap-2 items-center mb-4">
                        <span className="text-gray-300 text-sm">Overall</span>
                        <span className="text-red-400 text-sm font-medium">-5%</span>
                      </div>
                      
                      {/* Bar Chart */}
                      <div className="h-32 flex items-end justify-center gap-4">
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '80%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">A</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '50%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">B</span>
                        </div>
                        <div className="flex flex-col items-center">
                          <div className="w-8 bg-purple-500 border-t-2 border-purple-400" style={{height: '100%'}}></div>
                          <span className="text-xs text-gray-400 mt-2">C</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Model Retraining */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-4">Model Retraining (Optional)</h3>
                  <p className="text-white text-base mb-6">
                    Recommended improvements: Optimize hyperparameters for better performance.
                  </p>
                  
                  <div className="mb-6">
                    <button className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-all font-medium">
                      Retrain with optimized hyperparameters
                    </button>
                  </div>

                  {/* Configuration Table */}
                  <div className="overflow-hidden rounded-lg border border-gray-600">
                    <table className="w-full">
                      <thead>
                        <tr className="bg-gray-700/50">
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Configuration</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Hyperparameter A</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Hyperparameter B</th>
                          <th className="px-4 py-3 text-left text-white text-sm font-medium">Result</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 1</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 1</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 2</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">90%</td>
                        </tr>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 2</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 3</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 4</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">91%</td>
                        </tr>
                        <tr className="border-t border-gray-600">
                          <td className="px-4 py-3 text-white text-sm">Config 3</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 5</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">Value 6</td>
                          <td className="px-4 py-3 text-gray-300 text-sm">92%</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Export & Share */}
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-white mb-6">Export & Share</h3>
                  <div className="flex gap-4">
                    <button className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium">
                      Download PDF Report
                    </button>
                    <button className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700 transition-all font-medium">
                      Share Dashboard
                    </button>
                  </div>
                </div>
              </div>
                )}
              </>
            )}

            {activeTab === 'predict' && (
              <>
                {!selectedDataset && !uploadedFile ? (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-12 border border-gray-700 text-center">
                    <div className="max-w-md mx-auto">
                      <BarChart3 className="w-16 h-16 text-gray-500 mx-auto mb-6" />
                      <h2 className="text-2xl font-semibold text-white mb-4">Dataset Required</h2>
                      <p className="text-gray-300 mb-8">
                        To make predictions, you need to first have a dataset and train a model.
                      </p>
                      <div className="flex gap-4 justify-center">
                        <button
                          onClick={() => setActiveTab('search')}
                          className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-all font-medium"
                        >
                          Search Datasets
                        </button>
                        <button
                          onClick={() => setActiveTab('upload')}
                          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-all font-medium"
                        >
                          Upload Data
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                    <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                      <BarChart3 className="w-6 h-6 mr-3 text-orange-400" />
                      Make Predictions
                    </h2>
                    <PredictionForm />
                  </div>
                )}
              </>
            )}

            {activeTab === 'export' && (
              <>
                {!selectedDataset && !uploadedFile ? (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-12 border border-gray-700 text-center">
                    <div className="max-w-md mx-auto">
                      <FileText className="w-16 h-16 text-gray-500 mx-auto mb-6" />
                      <h2 className="text-2xl font-semibold text-white mb-4">Dataset Required</h2>
                      <p className="text-gray-300 mb-8">
                        To export models and deploy APIs, you need to first have a dataset and train a model.
                      </p>
                      <div className="flex gap-4 justify-center">
                        <button
                          onClick={() => setActiveTab('search')}
                          className="bg-purple-600 text-white px-6 py-3 rounded-lg hover:bg-purple-700 transition-all font-medium"
                        >
                          Search Datasets
                        </button>
                        <button
                          onClick={() => setActiveTab('upload')}
                          className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-all font-medium"
                        >
                          Upload Data
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-800/50 backdrop-blur-sm rounded-lg shadow-xl p-6 border border-gray-700">
                    <h2 className="text-2xl font-semibold mb-6 flex items-center text-white">
                      <FileText className="w-6 h-6 mr-3 text-cyan-400" />
                      Export & Deploy
                    </h2>
                    <ModelList 
                      onModelSelect={(modelId) => {
                        console.log('Selected model:', modelId)
                      }}
                    />
                  </div>
                )}
              </>
            )}
          </div>
        </main>

        {/* Bottom Training Logs Panel */}
        <div className="border-t border-gray-700 bg-gray-800/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white flex items-center">
                <Settings className="w-5 h-5 mr-2 text-gray-400" />
                Training Logs
              </h3>
              <div className="flex space-x-4">
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Download className="w-4 h-4" />
                </button>
                <button className="text-gray-400 hover:text-white transition-colors">
                  <Settings className="w-4 h-4" />
                </button>
              </div>
            </div>
            <div className="bg-gray-900 rounded-md p-4 h-32 overflow-y-auto border border-gray-700 font-mono text-sm">
              {logs.length === 0 ? (
                <p className="text-gray-400">No logs yet. Start training to see real-time progress...</p>
              ) : (
                <div className="space-y-1">
                  {logs.map((log, index) => (
                    <p key={index} className="text-green-400">
                      <span className="text-gray-500">[{new Date().toLocaleTimeString()}]</span> {log}
                    </p>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
