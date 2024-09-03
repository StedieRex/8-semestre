#include <16F877A.h>
#fuses XT, NOWDT
#use delay (clock = 4000000)

// Configuración del puerto A
#BYTE TRISA = 0X85
#BYTE PORTA = 0X05

// Configuración del puerto B
#BYTE TRISB = 0X86
#BYTE PORTB = 0X06

#BYTE OPTION_REG = 0X81

void main()
{
   // Configuración del puerto A
   bit_clear(OPTION_REG, 7);
   bit_set(TRISA, 0);     // RA0 como entrada
   bit_set(TRISA, 1);     // RA1 como entrada
   bit_clear(PORTA, 0);   // Asegura que RA0 está en bajo
   bit_clear(PORTA, 1);   // Asegura que RA1 está en bajo
   
   // Configuración del puerto B
   bit_clear(TRISB, 0);   // RB0 como salida
   bit_clear(PORTB, 0);   // Asegura que RB0 está en bajo

   while(1){
      if(bit_test(PORTA, 0) == 1 || bit_test(PORTA, 1) == 1)
         bit_set(PORTB, 0);   // Si RA0 o RA1 están en alto, enciende RB0
      else
         bit_clear(PORTB, 0); // Si ambos están en bajo, apaga RB0
   }  
}


