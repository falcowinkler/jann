(ns jann.sgd-test
  (:require [clojure.test :refer :all]
            [jann.network :as net]
            [jann.sgd :as sgd]
            [jann.initialization :as init])
  (:use [clojure.core.matrix]))

(def test-network (net/make-network [2 3 2] init/random-init))

;this can be considered a full "smoke test" for the network
;the dataset is just the domain of the or function (two one-hot encoded classes "true" and "false")
;and the network should be able to learn that in a reasonable amount of training episodes

(def or-dataset {:test   {:images [[1.0
                                    0.0]
                                   [1.0
                                    1.0]
                                   [0.0
                                    1.0]
                                   [0.0
                                    0.0]]
                          :labels [[1.0 0.0]
                                   [1.0 0.0]
                                   [1.0 0.0]
                                   [0.0 1.0]]}
                  :train {:images [[1.0
                                    0.0]
                                   [1.0
                                    1.0]
                                   [0.0
                                    1.0]
                                   [0.0
                                    0.0]]
                          :labels [[1.0 0.0]
                                   [1.0 0.0]
                                   [1.0 0.0]
                                   [0.0 1.0]]}})

(deftest test-xor
  (testing "if network is able to learn the or function through SGD"
    (is (= "4/4 correct classifications"
           (net/evaluate (get-in or-dataset [:test :images])
                         (get-in or-dataset [:test :labels])
                         (sgd/train or-dataset 4 test-network 100 3))))))