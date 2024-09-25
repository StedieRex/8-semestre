package prograDistribuida.algoritmosEleccion.gloton;
import java.util.Arrays;
import java.util.Comparator;

public class MochilaFraccionaria {
    public static double obtenerValorMaximo(int capacidad, Item[] items){
        Arrays.sort(items,new Comparator<Item>(){
            @Override
            public int compare(Item a, Item b){
                double realcion1 = (double)b.valor/b.peso;
                double realcion2 = (double)a.valor/a.peso;
                return Double.compare(realcion1,realcion2);
            }
        });
        double valorTotal = 0.0;
        for(Item item:items){
            if(capacidad-item.peso>=0){
                capacidad-=item.peso;
                valorTotal+=item.valor;
            }else{
                double fraccion = (double)capacidad/item.peso;
                valorTotal+=item.valor*fraccion;
                capacidad = 0;
                break;
            }
        }
        return valorTotal;
    }
    
    public static void main(String[] args) {
        Item[] items = {
            new Item(10,60),
            new Item(20,100),
            new Item(30,120)
        };

        int capacidadMochila = 50;
        double valorMaximo = obtenerValorMaximo(capacidadMochila,items);
        System.out.println("El valor maximo que se puede obtener es: "+valorMaximo);
    }
}
