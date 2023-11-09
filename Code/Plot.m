weeks = 0:48;
cases = [3, 2, 7, 12, 9, 10, 27, 21, 36, 63, 108, 255, 472, 675, 580, 844, 974, 1096, 1354, 1335, 1109, 936, 627, 476, 295, 164, 94, 37, 26, 15, 8, 5, 3, 1, 2, 0, 2, 1, 6, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0];
scatter(weeks, cases, 'r*');
title('The data of infected cases');
xlabel('Weeks');
ylabel('Infectious cases');