% Plik wygenerowany dynamicznie
:- style_check(-singleton).
contains_substring(String, Substring) :- sub_string(String, _, _, _, Substring).

policy_violation(Content, 'Odpowiedź nie zawiera przeprosin.') :- not( (contains_substring(Content, 'przepraszam') ; contains_substring(Content, 'przykro nam')) ).
policy_violation(Content, 'Odpowiedź obiecuje konkretną datę dostawy, co jest niedozwolone.') :- contains_substring(Content, 'będzie jutro').
policy_violation(Content, 'Odpowiedź nie oferuje klientowi żadnej formy rekompensaty (kuponu/rabatu).') :- not( (contains_substring(Content, 'rabat') ; contains_substring(Content, 'kupon')) ).