package supervisedMB;

public class Stats {
    public int numProfiles;
    public double blockSizes;

    public Stats(int numProfiles, double blockSizes){
        this.blockSizes = blockSizes;
        this.numProfiles = numProfiles;
    }
}
