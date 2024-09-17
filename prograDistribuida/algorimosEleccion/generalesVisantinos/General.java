import java.util.Random;

class General{
    private int id;
    private boolean isTraitor;
    private String decision;

    public General(int id,boolean isTraitor){
        this.id = id;
        this.isTraitor = isTraitor;
        this.decision = isTraitor ? randomDecision() : "ATTACK";//los traidores toman decisiones aleatorias
    }

    public String getDecision(){
        return decision;
    }

    public int getId(){
        return id;
    }

    public boolean isTraitor(){
        return isTraitor;
    }

    private String randomDecision(){
        Random rand = new Random();
        return rand.nextBoolean()?"ATTACK":"RETREAT";
    }
}