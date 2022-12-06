const DEFAULT_GRAYSCALE_WEIGHT = [0.2126, 0.7152, 0.0722]
const DEFAULT_GAUSSIAN_DISTRIBUTION_SIGMA = 1
const DEFAULT_GAUSSIAN_KERNEL_SIZE = 5
const SOBEL_OPERATOR_KERNEL_X = [
  [1, 0, -1],
  [2, 0, -2],
  [1, 0, -1]
]
const SOBEL_OPERATOR_KERNEL_Y = [
  [1, 2, 1],
  [0, 0, 0],
  [-1, -2, -1]
]
const DIRECTION = {
  EAST: 'EAST',
  SOUTH_EAST: 'SOUTH_EAST',
  SOUTH: 'SOUTH',
  SOUTH_WEST: 'SOUTH_WEST',
  WEST: 'WEST'
} as const
const DIRECTION_RANGE = {
  [DIRECTION.EAST]: [0, Math.PI / 8],
  [DIRECTION.SOUTH_EAST]: [Math.PI / 8, (Math.PI / 8) * 3],
  [DIRECTION.SOUTH]: [(Math.PI / 8) * 3, (Math.PI / 8) * 5],
  [DIRECTION.SOUTH_WEST]: [(Math.PI / 8) * 5, (Math.PI / 8) * 7],
  [DIRECTION.WEST]: [(Math.PI / 8) * 7, Math.PI]
}
const LOW_THRESHOLD = 30
const HIGH_THRESHOLD = 70

type GrayscaleMatrix = Array<Array<number>>

type GradientDirection = keyof typeof DIRECTION
type GradientDirections = Array<Array<GradientDirection>>

const hypot = (x: number, y: number) => {
  return Math.sqrt(x ** 2 + y ** 2)
}

const imageDataIndexToCoordinates = (imageData: ImageData, index: number) => {
  const i = index / 4

  return {
    x: i % imageData.width,
    y: Math.floor(i / imageData.width)
  }
}

const grayscaleMatrixToImageData = (grayscaleMatrix: GrayscaleMatrix) => {
  const h = grayscaleMatrix.length
  const w = grayscaleMatrix[0].length
  const imageData = new ImageData(new Uint8ClampedArray(w * h * 4), w, h)

  for (let i = 0; i < imageData.data.length; i += 4) {
    const { x, y } = imageDataIndexToCoordinates(imageData, i)

    for (let j = 0; j < 3; j++) {
      const n = i + j

      imageData.data[n] = Math.abs(grayscaleMatrix[y][x])
    }

    imageData.data[i + 3] = 255
  }

  return imageData
}

const cloneGrayscaleMatrix = (
  grayscaleMatrix: GrayscaleMatrix
): GrayscaleMatrix => {
  return [...new Array(grayscaleMatrix.length)].map((_, y) => {
    return [...new Array(grayscaleMatrix[0].length)].map((_, x) => {
      return grayscaleMatrix[y][x]
    })
  })
}

const createGrayscaleMatrix = (
  width: number,
  height: number
): GrayscaleMatrix => {
  return [...new Array(height)].map(() => {
    return [...new Array(width)].map(() => 0)
  })
}

const grayscale = (
  imageData: ImageData,
  weightCoefficients: Array<number>
): GrayscaleMatrix => {
  // * https://en.wikipedia.org/wiki/Grayscale

  const [rw, gw, bw] = weightCoefficients
  const grayscaleMatrix = createGrayscaleMatrix(
    imageData.width,
    imageData.height
  )

  for (let i = 0; i < imageData.data.length; i += 4) {
    const { x, y } = imageDataIndexToCoordinates(imageData, i)
    const value =
      imageData.data[i] * rw +
      imageData.data[i + 1] * gw +
      imageData.data[i + 2] * bw

    grayscaleMatrix[y][x] = value
  }

  return grayscaleMatrix
}

const gaussianDistribution2D = (x: number, y: number, sigma: number) => {
  // * https://homepages.inf.ed.ac.uk/rbf/HIPR2/gsmooth.htm

  const DSS = 2 * sigma ** 2 // Double Squared Sigma

  return (1 / (Math.PI * DSS)) * Math.exp(-(x ** 2 + y ** 2) / DSS)
}

const createGaussianKernel = (size: number, sigma: number) => {
  const k = Math.floor(size / 2)

  return [...new Array(size)].map((_, rowIndex) => {
    const y = rowIndex - k

    return [...new Array(size)].map((_, columnIndex) => {
      const x = columnIndex - k

      return gaussianDistribution2D(x, y, sigma)
    })
  })
}

const getMatrixValue = (
  matrix: GrayscaleMatrix,
  x: number,
  y: number,
  defaultValue: number
) => {
  return matrix[y]?.[x] ?? defaultValue
}

const applyConvolutionMatrix = (
  grayscaleMatrix: GrayscaleMatrix,
  matrix: Array<Array<number>>
) => {
  // * https://en.wikipedia.org/wiki/Convolution

  const clonedMatrix = cloneGrayscaleMatrix(grayscaleMatrix)
  const k = Math.floor(matrix.length / 2)

  for (let y = 0; y < grayscaleMatrix.length; y++) {
    for (let x = 0; x < grayscaleMatrix[0].length; x++) {
      grayscaleMatrix[y][x] = 0

      for (let yi = 0; yi < matrix.length; yi++) {
        const offsetY = yi - k
        const my = y + offsetY

        for (let xi = 0; xi < matrix[0].length; xi++) {
          const offsetX = xi - k
          const mx = x + offsetX
          const value =
            getMatrixValue(clonedMatrix, mx, my, clonedMatrix[y][x]) *
            matrix[yi][xi]

          grayscaleMatrix[y][x] += value
        }
      }
    }
  }

  return grayscaleMatrix
}

const applyGaussianBlur = (
  grayscaleMatrix: GrayscaleMatrix,
  size: number,
  sigma: number
) => {
  // * https://en.wikipedia.org/wiki/Gaussian_filter

  applyConvolutionMatrix(grayscaleMatrix, createGaussianKernel(size, sigma))
}

const applySobelOperator = (grayscaleMatrix: GrayscaleMatrix) => {
  // * https://en.wikipedia.org/wiki/Sobel_operator

  const [gmx, gmy] = [...new Array(2)].map(() => {
    return cloneGrayscaleMatrix(grayscaleMatrix)
  })

  applyConvolutionMatrix(gmx, SOBEL_OPERATOR_KERNEL_X)
  applyConvolutionMatrix(gmy, SOBEL_OPERATOR_KERNEL_Y)

  const gradientDirections: GradientDirections = [
    ...new Array(grayscaleMatrix.length)
  ].map(() => {
    return [...new Array(grayscaleMatrix[0].length)]
  })

  for (let y = 0; y < grayscaleMatrix.length; y++) {
    for (let x = 0; x < grayscaleMatrix[0].length; x++) {
      grayscaleMatrix[y][x] = hypot(gmx[y][x], gmy[y][x])

      const getAngle = () => {
        const angle = Math.atan2(gmy[y][x], gmx[y][x])

        if (angle < 0) {
          return angle + Math.PI
        }

        return angle
      }

      const angle = getAngle()

      gradientDirections[y][x] = Object.entries(DIRECTION_RANGE).find(
        ([, range]) => {
          return angle >= range[0] && angle <= range[1]
        }
      )![0] as GradientDirection
    }
  }

  return { grayscaleMatrix, gradientDirections }
}

const applyEdgesThinning = (
  grayscaleMatrix: GrayscaleMatrix,
  gradientDirections: GradientDirections
) => {
  // * https://en.wikipedia.org/wiki/Edge_detection#Edge_thinning

  const clonedMatrix = cloneGrayscaleMatrix(grayscaleMatrix)

  for (let y = 0; y < clonedMatrix.length; y++) {
    for (let x = 0; x < clonedMatrix[0].length; x++) {
      const value = clonedMatrix[y][x]

      if (value === 0) {
        continue
      }

      const direction = gradientDirections[y][x]

      const getSiblingCoordinates = () => {
        switch (direction) {
          case DIRECTION.EAST:
          case DIRECTION.WEST:
            return [
              { x: x - 1, y },
              { x: x + 1, y }
            ]
          case DIRECTION.SOUTH_EAST:
            return [
              { x: x - 1, y: y - 1 },
              { x: x + 1, y: y + 1 }
            ]
          case DIRECTION.SOUTH:
            return [
              { x, y: y - 1 },
              { x, y: y + 1 }
            ]
          case DIRECTION.SOUTH_WEST:
            return [
              { x: x - 1, y: y + 1 },
              { x: x + 1, y: y - 1 }
            ]
          default:
            return [
              { x, y },
              { x, y }
            ]
        }
      }

      const siblingCoordinates = getSiblingCoordinates()
      const [prevValue, nextValue] = siblingCoordinates.map(({ x, y }) => {
        return getMatrixValue(clonedMatrix, x, y, value)
      })

      if (value < prevValue || value < nextValue) {
        grayscaleMatrix[y][x] = 0
        continue
      }

      grayscaleMatrix[y][x] = value
    }
  }
}

const applyDoubleThreshold = (
  grayscaleMatrix: GrayscaleMatrix,
  low: number,
  high: number
) => {
  for (let y = 0; y < grayscaleMatrix.length; y++) {
    for (let x = 0; x < grayscaleMatrix[0].length; x++) {
      const value = Math.abs(grayscaleMatrix[y][x])

      if (value === 0) {
        continue
      }

      if (value < low) {
        grayscaleMatrix[y][x] = 0
        continue
      }

      if (value < high) {
        grayscaleMatrix[y][x] = 128
        continue
      }

      grayscaleMatrix[y][x] = 255
    }
  }
}

const applyHysteresis = (grayscaleMatrix: GrayscaleMatrix) => {
  // * https://en.wikipedia.org/wiki/Connected-component_labeling

  const clonedMatrix = cloneGrayscaleMatrix(grayscaleMatrix)

  const checkSiblings = (x: number, y: number) => {
    for (let yi = 0; yi < 3; yi++) {
      const offsetY = yi - 1
      const my = y + offsetY

      for (let xi = 0; xi < 3; xi++) {
        const offsetX = xi - 1
        const mx = x + offsetX

        if (mx === x && my === y) {
          continue
        }

        const value = getMatrixValue(clonedMatrix, mx, my, clonedMatrix[y][x])

        if (value !== 128) {
          continue
        }

        clonedMatrix[my][mx] = 255
        grayscaleMatrix[y][x] = 255

        checkSiblings(mx, my)
      }
    }
  }

  for (let y = 0; y < clonedMatrix.length; y++) {
    for (let x = 0; x < clonedMatrix[0].length; x++) {
      if (clonedMatrix[y][x] !== 255) {
        grayscaleMatrix[y][x] = 0

        continue
      }

      checkSiblings(x, y)
    }
  }
}

const canny = (canvas: HTMLCanvasElement) => {
  // * https://en.wikipedia.org/wiki/Canny_edge_detector

  const ctx = canvas.getContext('2d')!
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const result: Array<[ImageData, string]> = []

  const gm1 = grayscale(imageData, DEFAULT_GRAYSCALE_WEIGHT)

  result.push([grayscaleMatrixToImageData(gm1), 'grayscale'])

  applyGaussianBlur(
    gm1,
    DEFAULT_GAUSSIAN_KERNEL_SIZE,
    DEFAULT_GAUSSIAN_DISTRIBUTION_SIGMA
  )

  result.push([grayscaleMatrixToImageData(gm1), 'gaussian blur'])

  const { grayscaleMatrix: gm2, gradientDirections } = applySobelOperator(gm1)

  result.push([grayscaleMatrixToImageData(gm2), 'sobel operator'])

  applyEdgesThinning(gm2, gradientDirections)

  result.push([grayscaleMatrixToImageData(gm2), 'edge thinning'])

  applyDoubleThreshold(gm2, LOW_THRESHOLD, HIGH_THRESHOLD)

  result.push([grayscaleMatrixToImageData(gm2), 'double threshold'])

  applyHysteresis(gm2)

  result.push([grayscaleMatrixToImageData(gm2), 'edge tracking'])

  return result
}

export { canny }
