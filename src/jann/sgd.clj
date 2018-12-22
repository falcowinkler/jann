(ns jann.sgd
  (:require [jann.backpropagation :as backprop]
            [jann.dataset :as dataset]
            [jann.network :as net])
  (:use [clojure.core.matrix]
        [clojure.core.matrix.selection]))

(defn get-stochastic-gradient [m-images m-labels network]
  (loop [imgs m-images
         lbls m-labels
         nabla_b (for [s (:biases network)] (zero-array (shape s)))
         nabla_w (for [s (:weights network)] (zero-array (shape s)))]
    (if (empty? imgs)
      {:nabla_w nabla_w :nabla_b nabla_b}
      (let [{n_b :nabla_b n_w :nabla_w}
            (backprop/get-gradient
              (:weights network)
              (:biases network)
              (first imgs)
              (first lbls))]
        (recur (rest imgs)
               (rest lbls)
               (for [[x y] (map vector nabla_b n_b)] (add x y))
               (for [[x y] (map vector nabla_w n_w)] (add x y)))))))

(defn add-bias-to-nabla [batch-size learning-rate bias-nabla-pair]
  (let [averaged-nabla (mul (/ learning-rate batch-size) (second bias-nabla-pair))]
    (sub (first bias-nabla-pair) averaged-nabla)))

(defn apply-grad [grad network batch-size learning-rate]
  {:biases  (map (partial add-bias-to-nabla batch-size learning-rate)
                 (map vector (:biases network) (:nabla_b grad)))
   :weights (map (partial add-bias-to-nabla batch-size learning-rate)
                 (map vector (:weights network) (:nabla_w grad)))})

(defn train-epoch [images labels batch-size network learning-rate]
  (loop [index 0 net network]
    (let [m-images (subvec images index (+ index batch-size))
          m-labels (subvec labels index (+ index batch-size))
          grad (get-stochastic-gradient m-images m-labels net)
          updated-net (apply-grad grad net batch-size learning-rate)]
      (if (> (+ index (* 2 batch-size)) (count images))
        updated-net
        (recur (+ index batch-size) updated-net)))))

(defn train [dataset batch-size network epochs learning-rate]
  (loop [epoch 0 net network dset dataset]
    (println (net/evaluate
               (get-in dset [:test :images])
               (get-in dset [:test :labels]) net))
    (if (= epoch epochs)
      net
      (do
        (println "Training epoch " epoch)
        (recur (inc epoch)
               (train-epoch (get-in dset [:train :images])
                            (get-in dset [:train :labels])
                            batch-size net learning-rate)
               (dataset/shuffle-dataset dset))))))