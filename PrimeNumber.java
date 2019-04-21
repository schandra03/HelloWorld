public class PrimeNumber {

	public static void main(String args[]) {
		int start = 1;
		int end = 20;
		
		System.out.println("Prime Numbers between 1 to 20");
		
		while(start <= end) {
			int count = 0;
			for(int j=1; j <= start; j++) {
				if(start%j == 0)
					count ++;
			}
			if(count <= 2) {
				System.out.print(start + " ");
			}
			start++;
		}
	}
}