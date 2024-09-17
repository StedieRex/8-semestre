import java.util.ArrayList;
import java.util.List;

public class ByzantineProblem {
    private List<General>generals;

    public ByzantineProblem(int numGenerals,int numTraitors){
        generals = new ArrayList<>();
        for(int i = 0; i < numGenerals; i++){
            generals.add(new General(i,i < numTraitors));
        }
    }
    
    public void executeConsenus(){
        for(General general : generals){
            if(!general.isTraitor()){
                System.out.println("General "+general.getId()+"ve las siguientes decisiones:");
                for(General otherGeneral : generals){
                    System.out.println("General "+otherGeneral.getId()+"decide "+otherGeneral.getDecision());
                }
            }
        }

        String finalDecision=getMajorityDecision();
        System.out.println("La decision final es: "+finalDecision);
    }

    private String getMajorityDecision(){
        int attackCount = 0;
        int retreatCount = 0;
        for(General general : generals){
            if(general.getDecision().equals("ATTACK")){
                attackCount++;
            }else{
                retreatCount++;
            }
        }
        return attackCount >= retreatCount ? "ATTACK" : "RETREAT";
    }
}
