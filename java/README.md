## Java code

This folder contains the part of experiments written in Java.

To recompile them:
1. Download and configure [Maven](https://maven.apache.org/).
2. Open the shell and move into the `java_source` folder.
3. run the command `mvn clean package`. After the execution, there will be a new folder named `target`. Replace the `supMB.jar` with the one contained in the `target` folder (named `supMB-1.0-SNAPSHOT.jar`). Finally, replace the content of the `lib` folder with those of the `target/supMB-1.0-SNAPSHOT.lib` folder.