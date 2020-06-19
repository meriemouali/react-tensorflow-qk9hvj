import React, { Component } from 'react'
import { render } from 'react-dom'

import 'crypto-js'
import * as tf from '@tensorflow/tfjs'

class App extends Component {
  constructor() {
    super()
    this.state = {
      name: 'React'
    }
  }

  tensor = () => {
    console.log(`tensor`)
    const shape = [2, 3] // 2 rows, 3 columns
    const a = tf.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0], shape)
    a.print()    
  }

  tensorV2 = () => {
    console.log(`tensorV2`)
    const b = tf.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    b.print()
  }

  tensorSquare = () => {
    console.log(`tensorSquare`)
    const d = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]])
    const d_squared = d.square()
    d_squared.print()
  }

  tensorAdd = () => {
    console.log(`tensorAdd`)
    const e = tf.tensor2d([[1.0, 2.0], [3.0, 4.0]])
    const f = tf.tensor2d([[5.0, 6.0], [7.0, 8.0]])

    const e_plus_f = e.add(f)
    e_plus_f.print()    
  }

  model = () => {
    console.log(`model`)    
    const a = tf.scalar(2)
    const b = tf.scalar(4)
    const c = tf.scalar(8)

    function predict(input) {
      // y = a * x ^ 2 + b * x + c
      // More on tf.tidy in the next section
      return tf.tidy(() => {
        const x = tf.scalar(input)

        const ax2 = a.mul(x.square())
        const bx = b.mul(x)
        const y = ax2.add(bx).add(c)

        return y
      })
    }
    
    const result = predict(2)
    result.print()    
  }

  modelV2 = () => {
    console.log(`modelV2`)
    const model = tf.sequential()
    model.add(
      tf.layers.simpleRNN({
        units: 20,
        recurrentInitializer: 'GlorotNormal',
        inputShape: [80, 4]
      })
    )

    const learningRate = 0.01
    const optimizer = tf.train.sgd(learningRate)
    model.compile({ optimizer, loss: 'categoricalCrossentropy' })
    model.fit({ x: data, y: labels })    
  }

  simpleRNN = () => {
    console.log(`simpleRNN`)
    
    const rnn = tf.layers.simpleRNN({units: 8, returnSequences: true})

    // Create an input with 10 time steps.
    const input = tf.input({shape: [10, 20]})
    const output = rnn.apply(input)

    console.log(JSON.stringify(output.shape))    
  }

  componentWillMount() {
    this.tensor()
    
    this.tensorV2()
    
    this.tensorSquare()
    
    this.tensorAdd()

    this.model()

    //this.modelV2()

    this.simpleRNN()
  }

  render() {
    return (
      <div>
        <p>
          Start editing to see some magic happen :)
        </p>
      </div>
    )
  }
}

render(<App />, document.getElementById('root'))
