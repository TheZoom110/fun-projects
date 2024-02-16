#include <stdio.h>

#define mukhya main
#define sankhya int
#define lekh printf
#define por scanf
#define thikana *
#define mone_rakh &
#define jog +
#define biyog -
#define gun *
#define bhag /
#define jodi if
#define nahole else
#define samaan_hoy ==
#define na_hoy !=
#define boro_hoy >
#define choto_hoy <
#define er_theke 
#define na !
#define thik true
#define bhul false
#define firiye_de return

sankhya mukhya() {
	sankhya ka, kha, ga, gha;
	lekh("Ekti sankhya dao (ka): ");
	por("%d", mone_rakh ka);
	lekh("Ar ekti sankhya dao (kha): ");
	por("%d", mone_rakh kha);
	jodi (ka boro_hoy kha er_theke) {
		lekh("\nka=%d er maan kha=%d er maan er theke boro.", ka, kha);
	} nahole jodi (ka choto_hoy kha er_theke) {
		lekh("\nka=%d er maan kha=%d er maan er theke choto.", ka, kha);
	} nahole {
		lekh("\nka=%d ar kha=%d er maan samaan.", ka, kha);
	}
	lekh("\nJog korle amra fire pabo: %d", ka jog kha);
	firiye_de 0;
}
