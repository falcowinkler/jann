(ns jann.core
  (:use [clojure.core.matrix]
        [clojure.core.matrix.random])
  (:require
    [jann.sgd :as sgd]
    [jann.dataset :as ds]
    [jann.network :as n]
    [jann.initialization :as init])
  (:gen-class))

(set-current-implementation :vectorz)

(defn -main
  [& _]
  (let [ds (ds/load-dataset)
        network (n/make-network [784 100 10] init/random-init)]
    (sgd/train ds 10 network 30 3)))