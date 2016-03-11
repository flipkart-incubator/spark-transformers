# fk-ml-lib
Library for exporting spark models in Java ecosystem.

Goal of this library is to :
* Provide a way to export models into a custom format which can be imported back into a java object.
* Provide a way to do model predictions in java ecosystem.

#Usage
```
//Train model in spark
LogisticRegressionModel lrmodel = new LogisticRegressionWithSGD().run(data.rdd());

//Export this model
byte[] exportedModel = ModelExporter.export(lrmodel, null);

//Import and get Transformer
Transformer transformer = ModelImporter.importAndGetTransformer(exportedModel);

//predict
double predicted = (double) transformer.transform(new Double[] {0.3, 0.5. 0.6});
```
For detailed usage see unit tests. 
https://github.com/Flipkart/fk-ml-lib/blob/master/adapters/src/test/java/com/flipkart/fdp/ml/adapter/LogisticRegressionBridgeTest.java

## Getting help
For help regarding usage, drop an email to fdp-ml-dev@flipkart.com
