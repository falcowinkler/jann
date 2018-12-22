(ns jann.network
  (:require [jann.nonlinearities :as nonlin])
  (:use [clojure.core.matrix]
        [jann.common]))

(defn construct-weight-matrix [units init-fn]
  (vec
    (if (< (count units) 2)
      nil
      (cons (init-fn (zero-matrix (first (rest units)) (first units)))
            (construct-weight-matrix (rest units) init-fn)))))

(defn make-network [units init-fn]
  {:biases  (vec (for [per-layer-neurons (rest units)]
                   (init-fn (zero-vector per-layer-neurons))))
   :weights (construct-weight-matrix units init-fn)})

(defn- inference [input weights biases]
  (if (< (count weights) 1)
    input
    (inference
      (nonlin/sigmoid
        (wx+b (first weights) input (first biases)))
      (rest weights) (rest biases))))

(defn infer [input {:keys [weights biases]}]
  (inference input weights biases))

(defn argmax [x]
  (key (apply max-key val (reduce merge (map-indexed hash-map x)))))

(defn evaluate [test-data test-labels network]
  (let [results (map (fn [x] (argmax (inference x (:weights network) (:biases network)))) test-data)
        zipped (map vector results (map argmax test-labels))]
    (str  (get (frequencies (map (fn [x] (= (first x) (second x))) zipped)) true)
          "/"
          (count test-data) " correct classifications")))

