(ns jann.dataset-test
  (:require
    [clojure.test :refer :all]
    [jann.dataset :as dataset])
  (:use [clojure.core.matrix]))


(deftest load-dataset-test
  (testing "if edn dataset is loaded correctly"
    (is (= {:test  {:images [[0.0
                              1.0
                              0.0]]
                    :labels [[0.0
                              1.0]]}
            :train {:images [[1.0
                              0.0
                              0.0]
                             [1.0
                              0.0
                              0.0]
                             [0.0
                              0.0
                              1.0]]
                    :labels [[1.0
                              0.0]
                             [0.0
                              1.0]
                             [0.0
                              1.0]]}}
           (dataset/load-dataset "t10k-images-test.edn" "t10k-labels-test.edn")))))