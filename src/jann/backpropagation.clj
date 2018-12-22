(ns jann.backpropagation
  (:require [jann.nonlinearities :as nonlin]
            [jann.error-functions :as e])
  (:use [jann.common]
        [clojure.core.matrix]))

(defn calc-logits [weights biases input]
  (vec
    (if (< (count weights) 1)
      nil
      (let [logits (wx+b (first weights) input (first biases))]
        (cons logits
              (calc-logits (rest weights) (rest biases) (nonlin/sigmoid logits)))))))

(defn calc-activations [logits]
  (vec (emap nonlin/sigmoid logits)))

(defn output-error [activations logits label]
  (mul
    (e/mse-derivative (last activations) label)
    (nonlin/sigmoid-derivative (last logits))))

(defn layer-errors [weights next-error activations]
  (if (<= (count activations) 2)
    nil
    (let [error (mul
                  (mmul (transpose (last weights)) next-error)
                  (nonlin/sigmoid-derivative (last (drop-last activations))))]
      (cons error (layer-errors (drop-last weights) error (drop-last activations))))))

(defn nabla-w [deltas activations]
  (let [last-activations (transpose (reshape-1 (last (drop-last activations))))]
    (reverse
      (vec
        (if (< (count deltas) 1)
          nil
          (do
            (cons (mmul (reshape-1 (last deltas)) last-activations)
                  (nabla-w (drop-last deltas) (drop-last activations)))))))))

(defn get-gradient [weights biases input label]
  (let [logits (calc-logits weights biases input)
        activations (vec (cons input (calc-activations logits)))
        output-delta (output-error (rest activations) logits label)
        deltas (vec (conj (vec (layer-errors weights output-delta activations)) output-delta))
        grad-w (nabla-w deltas activations)]
    {:nabla_w grad-w :nabla_b deltas}))