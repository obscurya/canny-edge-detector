import { canny } from './canny'

import './style.css'

const container = document.querySelector<HTMLDivElement>('.container')!
const sourceCanvas = container.querySelector<HTMLCanvasElement>('#source')!
const sourceCtx = sourceCanvas.getContext('2d')!

const image = new Image()

image.onload = () => {
  sourceCanvas.width = image.width / 2
  sourceCanvas.height = image.height / 2

  sourceCtx.drawImage(image, 0, 0, sourceCanvas.width, sourceCanvas.height)

  canny(sourceCanvas).forEach(([imageData, text]) => {
    const canvasContainer = document.createElement('div')

    container.appendChild(canvasContainer)

    const canvas = document.createElement('canvas')

    canvas.width = sourceCanvas.width
    canvas.height = sourceCanvas.height

    const ctx = canvas.getContext('2d')!

    ctx.putImageData(imageData, 0, 0)
    canvasContainer.appendChild(canvas)

    const span = document.createElement('span')

    span.textContent = text

    canvasContainer.appendChild(span)
  })
}

// image.src = './chameleon.webp'

image.src = './image.jpg'
